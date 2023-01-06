"""Build the suffix automaton for a string."""

from copy import deepcopy
import torch
import logging
from tqdm import trange
import numpy as np

from .wfa import WFA
from .type_utils import to_int32
from .pysize import get_size


logger = logging.getLogger(__name__)
logger.setLevel(20)


class SuffixDfaBuilder:

  """Build the suffix automaton for each string.

  Follows the suffix automaton algorithm from Algorithms on Strings, page 205.

  Weights are 1-indexed pointers, where 0 represents null.
  ```
  """

  def __init__(self, dstore_size):
    n_states = 2 * dstore_size
    self.dfa = WFA(n_states, failures=True)

    # FIXME: use dstore_size appropriately here.

    self.L = -np.ones(n_states, dtype=np.int32)
    self.F = self.dfa.failures
    self.initial = None
    self.last = None

    self.dfa.solid_states = -np.ones(dstore_size + 1, dtype=np.int32)
    self.solid_states = self.dfa.solid_states

  def build(self, string: str):
    self.dfa.use_failures(False)
    self.initial = self.dfa.add_state(0)
    self.L[0] = 0
    self.F[self.initial] = -1  # Special convention of Algo.
    self.last = self.initial
    self.solid_states[0] = self.initial

    # This was causing hanging, memory leak.
    # for ptr, token in enumerate(string):
    length = len(string)
    for ptr in (pbar := trange(length)):
      token = string[ptr]
      token = to_int32(token)
      ptr = to_int32(ptr)
      self.extend(ptr, token)

    # Re-enable failures and quit.
    self.dfa.failures[self.initial] = self.initial
    self.dfa.use_failures(True)
    return self.dfa

  def extend(self, ptr, token):
    new = self.dfa.add_state(ptr + 1)
    self.L[new] = self.L[self.last] + 1
    self.solid_states[ptr] = new
    # self.L.append(self.L[self.last] + 1)
    # self.solid_states.append(new)
    state = self.last

    # Traverse failure path to 1) add edges to new state and 2) find first state with `token` transition.
    while True:
      self.dfa.add_edge(state, token, new)
      state = self.F[state]
      if state == -1:
        break
      next_state = self.dfa.next_state(state, token)
      if next_state != -1:
        break

    # No failure state contains token transition, i.e., no prefix of the current suffix exists as a factor.
    if state == -1:
      self.F[new] = self.initial

    # Can consult: https://en.wikipedia.org/wiki/Suffix_automaton

    # We found some factor that is a prefix of the current state.
    else:
      next_state = self.dfa.next_state(state, token)
      if self.L[state] + 1 == self.L[next_state]:
        self.F[new] = next_state

      # Clone the next state to make a smaller version of it.
      else:
        # weight = deepcopy(self.dfa.weights[next_state])
        # weight.append(ptr + 1)
        clone = self.dfa.add_state(-1)
        self.L[clone] = self.L[state] + 1
        # self.L.append(self.L[state] + 1)
        transitions = self.dfa.transitions[next_state]
        for s, q in transitions:
          self.dfa.add_edge(clone, s, q)
        # for s in self.dfa.edges_out[next_state]:
        #   q, w = self.dfa.transitions[next_state, s]
        #   self.dfa.add_edge(clone, s, q, w)
        self.F[new] = clone
        self.F[clone] = self.F[next_state]
        self.F[next_state] = clone

        # Traverse the failure path and relabel edges (next_state -> clone).
        next_state_ = next_state
        while True:
          # FIXME: If True, redirect arc, otherwise, add it.
          if next_state_ == next_state:
            self.dfa.remove_edge(state, token)
          # Indenting the following line fixes cabab test case.
          self.dfa.add_edge(state, token, clone)
          state = self.F[state]
          if state == -1:
            break
          next_state_ = self.dfa.next_state(state, token)
          if next_state != next_state_:
            break

    self.last = new
