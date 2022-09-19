"""Build the suffix automaton for a string."""

from copy import deepcopy

from .wfa import WFA
from .semiring import PointerSemiring


class SuffixAutomatonBuilder:

  """Build the suffix automaton for each string.

  Follows the suffix automaton algorithm from Algorithms on Strings, page 205.

  Operations can be chained. For example:

  ```python
  dfa = SuffixAutomatonBuilder().build("abbb").add_failures().dfa
  ```
  """

  def __init__(self):
    self.dfa = WFA(PointerSemiring())
    self.L = {}
    self.F = {}
    self.last = None
    # TODO: Could represent insolid states here: more memory efficient.
    # TODO: Instead, could do a binary array of size n.
    self.solid_states = []

  def build(self, string: str):
    initial = self.dfa.new_state(list(range(-1, len(string))))
    self.L[initial] = 0
    self.F[initial] = None
    self.last = initial
    self.solid_states.append(initial)

    for ptr, token in enumerate(string):
      self.extend(ptr, token)
    return self

  def extend(self, ptr, token):
    new = self.dfa.new_state([ptr])
    self.L[new] = self.L[self.last] + 1
    state = self.last
    self.solid_states.append(new)

    # Traversr failure path to 1) add edges to new state and 2) find first state with `token` transition.
    while True:
      self.dfa.add_edge(state, token, new, weight=None)
      state = self.F[state]
      next_state, _ = self.dfa.next_state(state, token)
      if state is None or next_state is not None:
        break

    # No failure state contains token transition, i.e., no prefix of the current suffix exists as a factor.
    if state is None:
      self.F[new] = self.dfa.initial

    # We found some factor that is a prefix of the current state.
    else:
      next_state, _ = self.dfa.next_state(state, token)
      if self.L[state] + 1 == self.L[next_state]:
        self.F[new] = next_state

      # Clone the next state to make a smaller version of it.
      else:
        # weight = self.dfa.weights[state]
        # weight = [x + 1 for x in weight]
        # weight = [ptr + 1]
        weight = self.dfa.weights[next_state] + [ptr]
        clone = self.dfa.new_state(weight)
        self.L[clone] = self.L[state] + 1
        for s in self.dfa.edges_out[next_state]:
          q, w = self.dfa.transitions[next_state, s]
          self.dfa.add_edge(clone, s, q, w)
        self.F[new] = clone
        self.F[clone] = self.F[next_state]
        self.F[next_state] = clone

        # Traverse the failure path and relabel edges (next_state -> clone).
        next_state_ = next_state
        while True:
          if next_state_ == next_state:
            self.dfa.remove_edge(state, token)
          # Indenting the following line fixes cabab test case.
          self.dfa.add_edge(state, token, clone, weight=None)
          state = self.F[state]
          next_state_, _ = self.dfa.next_state(state, token)
          if state is None or next_state != next_state_:
            break

    self.last = new

  def add_failures(self):
    """Add factors to a created DFA.

    This should only be called after `build()`.
    """
    self.dfa.failures = deepcopy(self.F)
    self.dfa.failures[self.dfa.initial] = self.dfa.initial
    return self
