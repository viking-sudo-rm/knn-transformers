import os
import numpy as np
import logging
import pickle

from .wfa import WFA


logger = logging.getLogger(__name__)


def flatten(li):
    return [item for sublist in li for item in sublist]


def get_flat_transitions(dfa: WFA):
    all_transitions = []
    lengths = []
    for transitions in dfa.transitions:
        flattened = flatten(transitions)
        all_transitions.extend(flattened)
        lengths.append(len(flattened))
    return np.array(all_transitions), np.array(lengths)


def get_transitions(flat_transitions, lengths):
    transitions = []
    cum_length = 0
    for length in lengths:
        flat_trans = flat_transitions[cum_length:cum_length + length]
        trans = [(flat_trans[idx], flat_trans[idx + 1]) for idx in range(0, len(flat_trans), 2)]
        transitions.append(trans)
        cum_length += length
    return transitions


def get_flat_weights(dfa: WFA):
    flat_weights = []
    lengths = []
    for ptrs in dfa.weights:
        flat_weights.extend(ptrs)
        lengths.append(len(ptrs))
    return np.array(flat_weights), np.array(lengths)


def get_weights(flat_weights, lengths):
    weights = []
    cum_length = 0
    for length in lengths:
        weight = list(flat_weights[cum_length:cum_length + length])
        weights.append(weight)
        cum_length += length
    return weights


def save_dfa(dir_path: str, dfa: WFA) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    logger.info("Saving transitions...")
    flat_transitions, transition_lengths = get_flat_transitions(dfa)
    np.save(os.path.join(dir_path, "flat_transitions.npy"), flat_transitions)
    np.save(os.path.join(dir_path, "transition_lengths.npy"), transition_lengths)

    if dfa.failures is not None:
        logger.info("Saving failures...")
        failures = np.array(dfa.failures)
        np.save(os.path.join(dir_path, "failures.npy"), failures)

    logger.info("Saving weights...")
    flat_weights, weight_lengths = get_flat_weights(dfa)
    np.save(os.path.join(dir_path, "flat_weights.npy"), flat_weights)
    np.save(os.path.join(dir_path, "weight_lengths.npy"), weight_lengths)

    if hasattr(dfa, "solid_states"):
        logger.info("Saving solid_states...")
        solid_states = np.array(dfa.solid_states)
        np.save(os.path.join(dir_path, "solid_states.npy"), solid_states)

    # TODO: Datatypes for tokens?
    metadata = {
        "sr": dfa.sr,
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
        failures = list(np.load(failures_path))
    else:
        failures = None

    flat_weights = np.load(os.path.join(dir_path, "flat_weights.npy"))
    weight_lengths = np.load(os.path.join(dir_path, "weight_lengths.npy"))
    weights = get_weights(flat_weights, weight_lengths)

    with open(os.path.join(dir_path, "metadata.p"), "rb") as fh:
        metadata = pickle.load(fh)

    dfa = WFA(metadata["sr"], failures=metadata["_failures"])
    dfa.initial = metadata["initial"]
    dfa.transitions = transitions
    dfa.failures = failures
    dfa.weights = weights

    solid_path = os.path.join(dir_path, "solid_states.npy")
    if os.path.exists(solid_path):
        dfa.solid_states = list(np.load(solid_path))

    return dfa
