from unittest import TestCase
import torch

from src.wfa import WFA
from src import state_lm
from src.trie_builder import TrieBuilder


dfa = WFA(2)
dfa.add_state()
dfa.add_state()
dfa.add_edge(0, 0, 0)
dfa.add_edge(0, 1, 1)
dfa.add_edge(1, 0, 1)


class StateLmTest(TestCase):

    def test_get_log_prob(self):
        lm = state_lm.StateLm(dfa, 2)
        states = torch.LongTensor([0, 0, 1])
        neg_dists = torch.tensor([0., -1000., 0.])
        log_dist = lm.get_log_prob(states, neg_dists)
        # State 0 is 50/50; state 1 predicts 1
        torch.testing.assert_close(log_dist.exp(), torch.tensor([.75, .25]))

    def test_get_log_prob_memoize(self):
        lm = state_lm.StateLm.create_memoized(dfa, 2)
        states = torch.LongTensor([0, 0, 1])
        neg_dists = torch.tensor([0., -1000., 0.])
        log_dist = lm.get_log_prob(states, neg_dists)
        # State 0 is 50/50; state 1 predicts 1
        torch.testing.assert_close(log_dist.exp(), torch.tensor([.75, .25]))
        self.assertEqual(lm.transitions.to_dense().tolist(), [[True, True], [True, False]])

    def test_unused_vocab_memoize(self):
        lm = state_lm.StateLm.create_memoized(dfa, 3)
        states = torch.LongTensor([0, 0, 1])
        neg_dists = torch.tensor([0., -1000., 0.])
        log_dist = lm.get_log_prob(states, neg_dists)
        torch.testing.assert_close(log_dist.exp(), torch.tensor([.75, .25, 0.]))
        self.assertEqual(lm.transitions.to_dense().tolist(), [[True, True, False], [True, False, False]])

    def test_unused_states_memoize(self):
        dfa = WFA(2)
        dfa.add_state()
        dfa.add_edge(0, 0, 0)
        dfa.add_edge(0, 1, 1)
        lm = state_lm.StateLm.create_memoized(dfa, 2)
        states = torch.LongTensor([0, 0, 1])
        neg_dists = torch.tensor([0., -1000., 0.])
        log_dist = lm.get_log_prob(states, neg_dists)
        # Probability mass is missing from state 1, since it has no outgoing transitions.
        torch.testing.assert_close(log_dist.exp(), torch.tensor([.25, .25]))
        self.assertEqual(lm.transitions.to_dense().tolist(), [[True, True], [False, False]])

    def test_on_chain(self):
        dfa = TrieBuilder(4).build(torch.tensor([4, 5, 4, 5]))
        lm = state_lm.StateLm(dfa, 10)
        states = torch.LongTensor([0, 1, 2, 3])
        neg_dists = torch.tensor([0., 0., -1000., 0.])
        log_dist = lm.get_log_prob(states, neg_dists)
        torch.testing.assert_close(log_dist, torch.tensor(
            [-1.0000e+04, -1.0000e+04, -1.0000e+04, -1.0000e+04, -1.0986e+00,
             -4.0547e-01, -1.0000e+04, -1.0000e+04, -1.0000e+04, -1.0000e+04]), atol=1e-04, rtol=1e-04)
