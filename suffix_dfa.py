from collections import defaultdict
import os

import logging
import pickle
import time
import numpy as np
import torch
from torch import nn
from enum import Enum, auto
from pathlib import Path
import glob
from itertools import zip_longest, islice, chain

from tqdm import tqdm

from knnlm import KNNWrapper, get_dstore_path

import faiss
import faiss.contrib.torch_utils
from faiss import IndexFlatL2
import scipy.sparse as sp

from src.suffix_dfa_builder import SuffixDfaBuilder
from src.trie_builder import TrieBuilder
from src.retriever import Retriever
from src.retriever_builder import RetrieverBuilder
from src.cache import Cache
from src.state_manager import StateManager

logger = logging.getLogger(__name__)
logger.setLevel(20)


class _Metrics:

    def __init__(self,
                 n_total: int = 0,
                 n_pointers: int = 0,
                 n_empty: int = 0,
                 n_states: int = 0,
                ):
        self.n_total = n_total
        self.n_pointers = n_pointers
        self.n_empty = n_empty
        self.n_states = n_states

    def update(self, sm, pointers) -> None:
        self.n_total += 1
        self.n_pointers += len(pointers)
        self.n_states += len(sm.states)
        if len(sm.states) == 0:
            self.n_empty += 1
    
    def get_metrics_dict(self) -> dict:
        return {
            "n_states": self.n_pointers / self.n_total,
            "n_pointers": self.n_pointers / self.n_total,
            "n_empty": self.n_empty / self.n_total,
        }


class SuffixDfaWrapper(KNNWrapper):
    def __init__(self,
                 no_pointer: bool = False,
                 min_knns: int = 1,
                 max_knns: int = 1024,
                 members=None,
                 truncate_dstore: int = -1,
                 min_factor_length: int = 2,
                 add_initial: bool = True,
                 cache_path: str = ".cache",
                 linear_dfa: bool = False,
                 solid_only: bool = False,
                 max_states: int = 1024,
                 **kwargs):
        super().__init__(**kwargs)
        self.no_pointer = no_pointer
        self.min_knns = min_knns
        self.max_knns = max_knns
        self.min_factor_length = min_factor_length
        self.add_initial = add_initial
        self.truncate_dstore = truncate_dstore
        self.cache_path = cache_path
        self.linear_dfa = linear_dfa
        self.solid_only = solid_only
        self.max_states = max_states
        self.metrics = _Metrics()

        if members is None:
            available_member_files = glob.glob(f'{self.dstore_dir}/members*')
            if len(available_member_files) == 0:
                logger.info(f'No member files found in {self.dstore_dir}, not using clustering')
            else:
                members = available_member_files[0]
                logger.info(f'Found the following cluster members files: {available_member_files}')
                logger.info(f'Using members file {members}')
        
        if members is None:
            self.extend_pointers_using_clusters = lambda pointers: pointers
        else:
            with open(members, 'rb') as file:
                self.members = pickle.load(file)
            members_for_indices = np.nonzero(self.members[np.arange(self.members.shape[0])])
            self.cluster = torch.zeros((self.dstore_size, ), dtype=torch.int32).to(self.device)
            self.cluster[members_for_indices[1]] = torch.from_numpy(members_for_indices[0]).to(self.device)

        self.generate_cur_knns = torch.tensor([], dtype=torch.int64)
        self.generate_cur_dists = torch.tensor([], dtype=torch.float32)
        self.no_lookup_counter_history = []

    def break_into(self, model):
        super().break_into(model)

        if not self.linear_dfa:
            cache_path = os.path.join(self.cache_path, str(self.truncate_dstore))
        else:
            cache_path = os.path.join(self.cache_path, f"{self.truncate_dstore}-reto")
        logger.info(f"Using cache in {cache_path}")
        cache = Cache(cache_path, log_fn=logger.info)
        cache.register_tuple(("dfa", "solid_states"), self._build_suffix_automaton)
        self.dfa = cache.get("dfa")
        self.solid_states = cache.get("solid_states")

        logger.info(f"# states: {len(self.dfa.weights)}")
        logger.info(f"# trans: {len(self.dfa.transitions)}")
        logger.info(f"# solid: {len(self.solid_states)}")
        logger.info(f"# fails: {len(self.dfa.failures)}")

        cache.register("inverse_failures", lambda: RetrieverBuilder.build_inverse_failures(self.dfa))
        cache.register("factor_lengths", lambda: RetrieverBuilder.build_factor_lengths(self.dfa))
        inverse_failures = cache.get("inverse_failures")
        factor_lengths = None if self.min_factor_length == 0 else cache.get("factor_lengths")
        self.retriever = Retriever(self.dfa, inverse_failures, factor_lengths, self.min_factor_length, max_pointers=None)

    def _build_suffix_automaton(self):
        """Build a suffix automaton over the data store, or load it if it is cached."""
        dstore = self.vals.reshape(-1)
        if self.truncate_dstore > -1:
            dstore = np.copy(dstore[:self.truncate_dstore])

        if not self.linear_dfa:
            builder = SuffixDfaBuilder()
            builder.build(dstore)
            logger.info("Adding failure transitions...")
            builder.add_failures()
            logger.info("Suffix DFA built!")
        else:
            builder = TrieBuilder()
            builder.build(dstore)
            logger.info("Linear DFA built!")
        return builder.dfa, builder.solid_states

    def post_forward_hook(self, module, input, output):
        shift = 0 if self.is_encoder_decoder else 1
        if self.labels is None:
            # In "generate" mode, we don't support yet tracking of the beam search hypotheses across time,
            # which we need to track in order to implement RetoMaton correctly. 
            # In the meantime, use kNN-LM's generate
            return super().post_forward_hook(module, input, output)

        lm_logits = output
        lm_logits = torch.nn.functional.log_softmax(lm_logits, dim=-1) # (batch, time, vocab)
        queries = self.activation_capturer.captured # (batch, time, dim) 
        
        shifted_labels = self.labels[:, shift:]
        nonpad_mask = torch.cat([
            shifted_labels != -100, 
            torch.zeros([self.labels.shape[0], shift], dtype=torch.bool).to(self.device)
        ], axis=-1)
        captured_labels = shifted_labels[shifted_labels != -100] # (nonpad)

        queries = queries[nonpad_mask] # (nonpad, dim)
        lm_logits = lm_logits[nonpad_mask] # (nonpad, vocab)

        all_knn_probs = []
        cur_knns = torch.tensor([], dtype=torch.int64)
        cur_dists = torch.tensor([], dtype=torch.float32)
        no_lookup_counter = 0

        sm = StateManager(self.solid_states,
                          self.retriever,
                          solid_only=self.solid_only,
                          max_states=self.max_states,
                          add_initial=self.add_initial)

        for idx, (timestep_query, label) in enumerate(zip_longest(queries, captured_labels)):
            perform_search = False
            pointers = torch.tensor(sm.get_pointers())
            self.metrics.update(sm, pointers)

            if self.no_pointer or pointers.numel() < self.min_knns:
                perform_search = True
                self.no_lookup_counter_history.append(no_lookup_counter)
                no_lookup_counter = 0
            else:
                no_lookup_counter += 1
            
            # (vocab_size, ) , (k, ), (k, ), (k, )
            cur_knn_log_prob, knns, dists, vals_at_knns = self.get_knn_log_prob(
                timestep_query, 
                pointers=pointers,
                perform_search=perform_search)
            all_knn_probs.append(cur_knn_log_prob)

            if not self.no_pointer and label is not None:
                vals_are_correct_and_pointer_available = (vals_at_knns == label) & (knns < self.dstore_size - 1)
                cur_knns = knns[vals_are_correct_and_pointer_available]
                cur_dists = dists[vals_are_correct_and_pointer_available]
                cur_knns = cur_knns[cur_dists.argsort(descending=True)]
                if self.truncate_dstore > -1:
                    cur_knns = cur_knns[cur_knns < self.truncate_dstore]
                # FIXME: This should remove active states.
                sm.add_pointers(cur_knns)

            # In the DFA, the token is an np.int32, but the type conversion should work out.
            sm.transition(token=label.item())

        interpolated_scores = KNNWrapper.interpolate(torch.stack(all_knn_probs), lm_logits, self.lmbda) # (nonpad, vocab)
        output[nonpad_mask] = interpolated_scores
        return output

    def get_knn_log_prob(self, query, pointers, perform_search):
        pointer_dists = torch.tensor([[]]).to(self.device)
        if pointers is not None and pointers.numel() > 0 and not self.recompute_dists:
            pointer_vectors = self.reconstruct_ids(pointers)
            pointer_dists = self.dist_func(query, pointer_vectors)
        
        if perform_search:
            dists, knns = self.get_knns(query.unsqueeze(0)) # (1, k)
            dists, knns = dists.squeeze(0), knns.squeeze(0) # (k, )
            if pointers is not None and pointers.numel() > 0:
                knns = torch.cat([knns, pointers], axis=-1)
                dists = torch.cat([dists, pointer_dists], axis=-1)
        else:
            knns = pointers
            dists = pointer_dists

        if self.recompute_dists:
            knns_vecs = torch.from_numpy(self.keys[knns]).to(self.device)
            dists = self.dist_func(query, knns_vecs) 
        
        neg_dists = -dists       
        knn_log_probs, vals_at_knns = self.knns_to_log_prob(knns, neg_dists)
        
        return knn_log_probs, knns, neg_dists, vals_at_knns

    def extend_pointers_using_clusters(self, pointers):
        if pointers.numel() == 0:
            return pointers
        # Don't take the same cluster twice
        # pointers = pointers.numpy()
        clusters, cluster_counts = torch.unique(self.cluster[pointers], return_counts=True)
        # Take smaller clusters first
        clusters = clusters[torch.argsort(-cluster_counts)]
        members = torch.from_numpy(np.nonzero(self.members[clusters.cpu().numpy()])[1]).to(self.device)
        # Prefer datastore entries that were directly pointed to by the previous time step's
        # datastore entries, over other members of their cluster
        extended_pointers = torch.cat([pointers, members])
        if len(extended_pointers) > self.max_knns:
            extended_pointers = extended_pointers[:self.max_knns]
        return extended_pointers

    def reconstruct_ids(self, ids):
        # Converting to numpy only because GPU indexes do not support reconstructing vectors:
        # https://github.com/facebookresearch/faiss/issues/2181
        ids = ids.cpu().numpy()
        # faiss's index.reconstruct supports a single ID at a time, so batching is performed
        # using numpy.vectorize:
        # https://github.com/facebookresearch/faiss/issues/1163
        reconstruct_func = np.vectorize(lambda x: self.reconstruct_index.reconstruct(int(x)), otypes=[object])
        vectors = reconstruct_func(ids)
        vectors = np.stack(vectors).reshape(ids.shape + (self.dimension, ))
        vectors_t = torch.from_numpy(vectors).to(self.device)
        return vectors_t

    def get_metrics(self):
        metrics = {'lookups_saved': np.sum(self.no_lookup_counter_history)/
            (np.sum(self.no_lookup_counter_history) + len(self.no_lookup_counter_history)),
        }
        metrics.update(self.metrics.get_metrics_dict())
        return metrics

    def break_out(self):
        super().break_out()
        self.print_stats()
    
    def print_stats(self):
        if len(self.no_lookup_counter_history) > 0:
            saved = np.sum(self.no_lookup_counter_history) / \
                (np.sum(self.no_lookup_counter_history) + len(self.no_lookup_counter_history))
            logger.info(f'Lookups saved: {saved*100}%')

    def cluster_dstore(self, num_clusters, sample_size, model, batch_size=500000):
        keys_vals_prefix = get_dstore_path(self.dstore_dir, model.config.model_type, self.dstore_size, self.dimension)
        keys = np.memmap(f'{keys_vals_prefix}_keys.npy', dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))

        if sample_size > self.dstore_size:
            logger.info('Taking all data for training')
            to_cluster = keys[:]
        else:
            idx = np.random.RandomState(1).choice(np.arange(self.dstore_size), size=sample_size, replace=False)
            to_cluster = keys[idx]

        to_cluster = to_cluster.astype(np.float32)
        logger.info(f'Starting to cluster {sample_size} examples into {num_clusters} clusters')
        kmeans = faiss.Kmeans(self.dimension, num_clusters, niter=20, verbose=True, gpu=True, seed=1)
        kmeans.train(to_cluster)

        logger.info(f'Done training, assigning {self.dstore_size} examples to the clusters')

        index = IndexFlatL2(self.dimension)
        index.add(kmeans.centroids)
        logger.info('Index created, added centroids')
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            logger.info('Moving index to GPU')
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index, co)
            logger.info('Moved index to GPU')

        start = 0
        centroid_ids = []
        logger.info('Starting to add tokens')
        while start < self.dstore_size:
            end = min(self.dstore_size, start + batch_size)
            to_search = keys[start:end].copy()
            _, key_i = index.search(torch.from_numpy(to_search).float(), 1)
            centroid_ids.append(key_i.squeeze())
            start += batch_size
            if (start % 1000000) == 0:
                print('Assigned %d tokens so far' % start)

        centroid_ids = np.concatenate(centroid_ids)

        logger.info('Processing the mapping of cluster->members')
        parent_cluster = centroid_ids
        cluster_to_members = defaultdict(set)
        for key_i, cluster in tqdm(enumerate(parent_cluster), total=self.dstore_size):
            cluster_to_members[cluster.item()].add(key_i)

        row_ind = [k for k, v in cluster_to_members.items() for _ in range(len(v))]
        col_ind = [i for ids in cluster_to_members.values() for i in ids]
        members_sp = sp.csr_matrix(([1]*len(row_ind), (row_ind, col_ind)))

        members_filename = get_members_path(self.dstore_dir, 
            model.config.model_type, self.dstore_size, self.dimension,
            sample_size, num_clusters)
        with open(members_filename, 'wb') as f:
            pickle.dump(members_sp, f)

        logger.info(f'Done, found {len(cluster_to_members)} clusters, written to {members_filename}')

def get_members_path(dstore_dir, model_type, dstore_size, dimension, sample_size, num_clusters):
    return f'{dstore_dir}/members_{model_type}_{dstore_size}_{dimension}_{sample_size}_{num_clusters}.pkl'
