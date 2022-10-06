"""Class representing weighted finite automata."""

import collections
from typing import Any, Dict, Iterable

from . import semiring


State = int
Token = str


class WFA:

  """Class representing a deterministic weighted automaton over some semiring.

  It is possible to modify the automaton, score strings, and merge states.
  """

  def __init__(self, sr: semiring.Semiring):
    self.sr = sr
    self.counter = 0
    self.initial = None
    self.weights = {}
    self.transitions = {}
    self.failures: Dict[State, State] = {}
    self.edges_out = collections.defaultdict(list)

  def next_state(self, state, token) -> tuple[State, Any]:
    """Return next state and transition weight of a token given the current state."""
    if (state, token) in self.transitions:
      return self.transitions[state, token]
    elif state in self.failures:
      fail_state = self.failures[state]
      if fail_state == state:
        # If we reach a cycle failure transition, we consume the current token.
        # This makes the DFA work correctly.
        return state, None
      return self.next_state(fail_state, token)
    else:
      return None, None
  
  def transition(self, string: Iterable[Token], state=None) -> tuple[State, Any]:
    state = state or self.initial
    cum_weight = self.sr.one
    for token in string:
      state, weight = self.next_state(state, token)
      if state is None:
        return state, cum_weight
      # Only update weight from transition if it is specified.
      if weight is not None:
        cum_weight = self.sr.mul(cum_weight, weight)
    return state, cum_weight

  def forward(self, string: Iterable[Token]) -> "self.sr.type":
    """Compute the weight assigned to string.

    Args:
      string: String to score with the WFA.

    Returns:
      Weight assigned to string by the WFA.
    """
    state, cum_weight = self.transition(string)
    if state is None or self.weights[state] is None:
      return self.sr.zero
    elif cum_weight is None or cum_weight == self.sr.one:
      return self.weights[state]
    else:
      return self.sr.mul(cum_weight, self.weights[state])

  def new_state(self, weight=None) -> State:
    assert weight is None or isinstance(weight, self.sr.type)
    state = self.counter
    self.weights[state] = weight
    self.counter += 1
    if self.initial is None:
      self.initial = state
    return state

  # FIXME(lambdaviking): Rewrite without adjacency representation.
  # def remove_state(self, state: State) -> None:
  #   del self.weights[state]
  #   for token, state2, weight in self.edges_in[state]:
  #     self.edges_out[state2].remove((token, state, weight))
  #     del self.transitions[state, token]
  #   del self.edges_in[state]
  #   for token, state2, weight in self.edges_out[state]:
  #     self.edges_in[state2].remove((token, state, weight))
  #   del self.edges_out[state]

  def add_edge(self, state1, token, state2, weight=None):
    assert weight is None or isinstance(weight, self.sr.type)
    self.transitions[state1, token] = (state2, weight)
    # We use these data structures to track which tokens come out of a state.
    self.edges_out[state1].append(token)
  
  def remove_edge(self, state: State, token: Token) -> bool:
    del self.transitions[state, token]
    return self.edges_out[state].remove(token)
