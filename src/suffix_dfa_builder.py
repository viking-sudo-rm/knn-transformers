"""Build the suffix automaton for a string."""

from copy import deepcopy
import torch
import logging
from tqdm import trange

from .wfa import WFA
from .semiring import PointerSemiring
from .type_utils import to_int32
from .pysize import get_size


logger = logging.getLogger(__name__)
logger.setLevel(20)


class SuffixDfaBuilder:

  """Build the suffix automaton for each string.

  Follows the suffix automaton algorithm from Algorithms on Strings, page 205.

  Operations can be chained. For example:

  ```python
  dfa = SuffixDfaBuilder().build("abbb").add_failures().dfa
  ```
  """

  def __init__(self):
    self.dfa = WFA(PointerSemiring(), failures=True)
    # TODO: Could represent insolid states here: more memory efficient.
    # TODO: Instead, could do a binary array of size n.
    self.solid_states = []

    self.L = []
    self.F = self.dfa.failures
    self.initial = None
    self.last = None

  def build(self, string: str):
    self.dfa.use_failures(False)
    self.initial = self.dfa.new_state([to_int32(0)])
    self.L.append(0)
    self.F[self.initial] = None  # Special convention of algo.
    self.last = self.initial
    self.solid_states.append(self.initial)

    # This was causing hanging, memory leak.
    # for ptr, token in enumerate(string):
    length = len(string)
    for ptr in (pbar := trange(length)):
      token = string[ptr]
      token = to_int32(token)
      ptr = to_int32(ptr)
      self.extend(ptr, token)
      # Quadratic in the size
      # if ptr % (length // 100) == 0:
      #   n_bytes = get_size(self.dfa)
      #   gbytes = n_bytes // 1024**3
      #   pbar.set_description(f"Size: {gbytes}M")

    # Re-enable failures and quit.
    self.dfa.failures[self.initial] = self.initial
    self.dfa.use_failures(True)
    self.dfa.solid_states = self.solid_states
    return self

  def extend(self, ptr, token):
    new = self.dfa.new_state([ptr + 1])
    self.L.append(self.L[self.last] + 1)
    state = self.last
    self.solid_states.append(new)

    # Traverse failure path to 1) add edges to new state and 2) find first state with `token` transition.
    while True:
      self.dfa.add_edge(state, token, new)
      state = self.F[state]
      if state is None:
        break
      next_state = self.dfa.next_state(state, token)
      if next_state is not None:
        break

    # No failure state contains token transition, i.e., no prefix of the current suffix exists as a factor.
    if state is None:
      self.F[new] = self.initial

    # Can consult: https://en.wikipedia.org/wiki/Suffix_automaton

    # We found some factor that is a prefix of the current state.
    else:
      next_state = self.dfa.next_state(state, token)
      if self.L[state] + 1 == self.L[next_state]:
        self.F[new] = next_state

      # Clone the next state to make a smaller version of it.
      else:
        weight = deepcopy(self.dfa.weights[next_state])
        weight.append(ptr + 1)
        clone = self.dfa.new_state(weight)
        self.L.append(self.L[state] + 1)
        transitions = self.dfa.transitions[next_state]
        for s, q in transitions:
          self.dfa.add_edge(clone, s, q)
        # self.L[clone] = self.L[state] + 1
        # for s in self.dfa.edges_out[next_state]:
        #   q, w = self.dfa.transitions[next_state, s]
        #   self.dfa.add_edge(clone, s, q, w)
        self.F[new] = clone
        self.F[clone] = self.F[next_state]
        self.F[next_state] = clone

        # Traverse the failure path and relabel edges (next_state -> clone).
        next_state_ = next_state
        while True:
          if next_state_ == next_state:
            self.dfa.remove_edge(state, token)
          # Indenting the following line fixes cabab test case.
          self.dfa.add_edge(state, token, clone)
          state = self.F[state]
          if state is None:
            break
          next_state_ = self.dfa.next_state(state, token)
          if next_state != next_state_:
            break

    self.last = new
