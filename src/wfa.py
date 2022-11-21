"""Class representing weighted finite automata."""

from typing import Any, Optional, Iterable

from .binary_search import binary_search
from . import semiring


State = int
Token = str


class WFA:

  """Class representing a deterministic weighted automaton over some semiring.

  It is possible to modify the automaton, score strings, and merge states.
  """

  def __init__(self, sr: semiring.Semiring, failures: bool = False):
    self.sr = sr
    self._failures = failures
    self.initial = None

    self.weights: list[int] = []
    self.transitions: list[list[int, int]] = []
    self.failures: Optional[list[int]]
    if failures:
      self.failures = []
    else:
      self.failures = None

  # We return a weight for API consistency.
  def next_state(self, state, token) -> State:
    """Return next state and transition weight of a token given the current state."""
    transitions = self.transitions[state]
    idx = binary_search(token, transitions)
    if idx < len(transitions):
      key, target = transitions[idx]
      if key == token:
        return target
    
    # Otherwise, we follow a failure transition.
    if not self._failures or self.failures is None:
      return None
    fail_state = self.failures[state]
    if fail_state == None:
      return None
    if fail_state == state:
      return state
    return self.next_state(fail_state, token)

  def transition(self, string: Iterable[Token], state=None) -> State:
    state = state or self.initial
    for token in string:
      state = self.next_state(state, token)
      if state is None:
        return None
    return state

  def forward(self, string: Iterable[Token]) -> "self.sr.type":
    """Compute the weight assigned to string.

    Args:
      string: String to score with the WFA.

    Returns:
      Weight assigned to string by the WFA.
    """
    state = self.transition(string)
    if state is None or self.weights[state] is None:
      return self.sr.zero
    else:
      return self.weights[state]

  def new_state(self, weight=None) -> State:
    state = len(self.weights)
    if self.initial is None:
      self.initial = state
    self.weights.append(weight)
    self.transitions.append([])
    if self.failures is not None:
      self.failures.append(None)
    return state

  def add_edge(self, state1, token, state2) -> bool:
    transitions = self.transitions[state1]
    idx = binary_search(token, transitions)
    if idx == len(transitions):
      transitions.insert(idx, (token, state2))
      return True
    key, target = transitions[idx]
    if key != token:
      transitions.insert(idx, (token, state2))
      return True
    if target == state2:
      return False
    transitions[idx] = (token, state2)
    return True
  
  def remove_edge(self, state: State, token: Token) -> bool:
    transitions = self.transitions[state]
    idx = binary_search(token, transitions)
    key, target = transitions[idx]
    if key == token:
      transitions.pop(idx)
      return True
    return False

  def use_failures(self, status: bool):
    self._failures = status
