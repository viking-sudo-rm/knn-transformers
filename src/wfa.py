"""Class representing a weighted automaton, where the weights can be int-valued (via Python int method)."""

from typing import Any, Optional, Iterable
import numpy as np

from .binary_search import binary_search


State = int
Pointer = int
Token = str


class WFA:

  """Class representing the automaton.

  Generally, states can be updated via next_state or transition.

  We also implement a forward method for ease of access, though this doesn't get called by the API.

  States are integers. -1 represents a failure sink state.

  Weights are integer-valued, representing pointers in the datastore. -1 is reserved for empty. 
  """

  def __init__(self, n_states: int, failures: bool = False):
    self.n_states = n_states
    self._failures = failures
    self.initial = None

    self.weights = -np.ones(n_states, dtype=np.int32)
    self.transitions: list[list[tuple[Token, State]]] = []

    if failures:
      self.failures = -np.ones(n_states, dtype=np.int32)
    else:
      self.failures = None

  # We return a weight for API consistency.
  def next_state(self, state, token) -> State:
    """Return next state and transition weight of a token given the current state."""
    if state == -1:
      return -1

    transitions = self.transitions[state]
    idx = binary_search(token, transitions)
    if idx < len(transitions):
      key, target = transitions[idx]
      if key == token:
        return target
    
    # Otherwise, we follow a failure transition.
    if not self._failures or self.failures is None:
      return -1
    fail_state = self.failures[state]
    if fail_state == -1:
      return -1
    if fail_state == state:
      return state
    return self.next_state(fail_state, token)

  def transition(self, string: Iterable[Token], state=None) -> State:
    state = state or self.initial
    for token in string:
      state = self.next_state(state, token)
      if state == -1:
        return -1
    return state

  def forward(self, string: Iterable[Token]) -> Pointer:
    """Compute the weight assigned to string.

    Args:
      string: String to score with the WFA.

    Returns:
      Weight assigned to string by the WFA.
    """
    state = self.transition(string)
    if state == -1:
      return -1
    else:
      return self.weights[state]

  def add_state(self, weight=-1) -> State:
    state = len(self.transitions)
    self.weights[state] = weight
    self.transitions.append([])
    if self.initial is None:
      self.initial = state
    if self.failures is not None:
      self.failures[state] = -1
    return state

  def add_edge(self, state1, token, state2) -> bool:
    transitions = self.transitions[state1]
    idx = binary_search(token, transitions)
    entry = (token, state2)
    if not transitions:
      transitions.append(entry)
      return True
    key, target = transitions[idx]
    if token < key:
      transitions.insert(idx, entry)
      return True
    elif token == key:
      transitions[idx] = entry
      return True
    else:
      transitions.insert(idx + 1, entry)
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
