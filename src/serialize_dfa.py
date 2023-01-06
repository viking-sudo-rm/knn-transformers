import os
import numpy as np
import logging
import pickle
import gc

from .wfa import WFA


logger = logging.getLogger(__name__)


def flatten(li):
    return (item for sublist in li for item in sublist)


def get_flat_transitions(dfa: WFA):
    all_transitions = []
    lengths = []
    for transitions in dfa.transitions:
        flattened = flatten(transitions)
        pre_length = len(all_transitions)
        all_transitions.extend(flattened)
        post_length = len(all_transitions)
        lengths.append(post_length - pre_length)
    return np.array(all_transitions), np.array(lengths)


def get_transitions(flat_transitions, lengths):
    transitions = []
    cum_length = 0
    for length in lengths:
        flat_trans = flat_transitions[cum_length:cum_length + length]
        trans = [(flat_trans[idx], flat_trans[idx + 1]) for idx in range(0, len(flat_trans), 2)]
        transitions.append(trans)
        cum_length += length

    del flat_transitions
    del lengths
    gc.collect()

    return transitions


def save_dfa(dir_path: str, dfa: WFA) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    logger.info("Saving transitions...")
    flat_transitions, transition_lengths = get_flat_transitions(dfa)
    np.save(os.path.join(dir_path, "flat_transitions.npy"), flat_transitions)
    np.save(os.path.join(dir_path, "transition_lengths.npy"), transition_lengths)

    if dfa.failures is not None:
        logger.info("Saving failures...")
        np.save(os.path.join(dir_path, "failures.npy"), dfa.failures)

    logger.info("Saving weights...")
    np.save(os.path.join(dir_path, "weights.npy"), dfa.weights)

    if hasattr(dfa, "solid_states"):
        logger.info("Saving solid_states...")
        np.save(os.path.join(dir_path, "solid_states.npy"), dfa.solid_states)

    # TODO: Datatypes for tokens?
    metadata = {
        "n_states": dfa.n_states,
        "_failures": dfa._failures,
        "initial": dfa.initial,
    }
    with open(os.path.join(dir_path, "metadata.p"), "wb") as fh:
        pickle.dump(metadata, fh)
    logger.info("Done!")


def load_dfa(dir_path: str):
    flat_transitions = np.load(os.path.join(dir_path, "flat_transitions.npy"))
    transition_lengths = np.load(os.path.join(dir_path, "transition_lengths.npy"))
    transitions = get_transitions(flat_transitions, transition_lengths)

    failures_path = os.path.join(dir_path, "failures.npy")
    if os.path.exists(failures_path):
        failures = np.load(failures_path)
    else:
        failures = None

    weights_path = os.path.join(dir_path, "weights.npy")
    weights = np.load(weights_path)

    with open(os.path.join(dir_path, "metadata.p"), "rb") as fh:
        metadata = pickle.load(fh)

    dfa = WFA(metadata["n_states"], failures=metadata["_failures"])
    dfa.initial = metadata["initial"]
    dfa.transitions = transitions
    dfa.failures = failures
    dfa.weights = weights

    solid_path = os.path.join(dir_path, "solid_states.npy")
    if os.path.exists(solid_path):
        dfa.solid_states = np.load(solid_path)

    return dfa
