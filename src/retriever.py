from collections import defaultdict
from typing import Iterable
import pickle


def _get_inverse_failures(dfa):
  inverse_failures = defaultdict(list)
  for state, fail_state in dfa.failures.items():
    if state is None or fail_state is None:
      continue
    inverse_failures[fail_state].append(state)
  inverse_failures[dfa.initial].remove(dfa.initial)
  return inverse_failures

def _get_factor_lengths(dfa):
  factor_lengths = {dfa.initial: 0}
  queue = [dfa.initial]
  while queue:
    state = queue.pop(0)
    length = factor_lengths[state]
    for token in dfa.edges_out[state]:
      next_state, _ = dfa.next_state(state, token)
      factor_lengths[next_state] = length + 1
      queue.append(next_state)
  return factor_lengths


class Retriever:

  """Retrieve contexts that are similar to the context."""

  def __init__(self,
               dfa,
               inverse_failures,
               factor_lengths,
               min_factor_length: int = 0,
               max_pointers: int = 1024,
              ):
    self.dfa = dfa
    self.inverse_failures = inverse_failures
    self.factor_lengths = factor_lengths
    self.min_factor_length = min_factor_length
    self.max_pointers = max_pointers

    self.n_retrievals = 0
    self.retrieved = defaultdict(int)

  @classmethod
  def create(cls, dfa, **kwargs):
    min_factor_length = kwargs["min_factor_length"]
    inverse_failures = _get_inverse_failures(dfa)
    factor_lengths = _get_factor_lengths(dfa) if min_factor_length > 0 else None
    return cls(dfa, inverse_failures, factor_lengths, **kwargs)

  @classmethod
  def load(cls, dfa, path, **kwargs):
    with open(path, "rb") as fh:
      cache = pickle.load(fh)
    inverse_failures = cache["inverse_failures"]
    factor_lengths = cache["factor_lengths"]
    return cls(dfa, inverse_failures, factor_lengths, **kwargs)
  
  def save(self, path: str) -> None:
    cache = {
      "inverse_failures": self.inverse_failures,
      "factor_lengths": self.factor_lengths,
    }
    with open(path, "wb") as fh:
      pickle.dump(cache, fh)

  def gen_pointers(self, states: Iterable[int]) -> Iterable[int]:
    """Get pointers out of a state in the DFA."""
    self.n_retrievals += 1
    counter = 0
    # TODO: Follow failure path of each state while >= self.min_factor_length??
    # TODO: Would this be different than a k-gram lookup model?
    queue = list(states)

    while self.max_pointers is None or counter < self.max_pointers:
      if not queue:
        break
      q = queue.pop(0)
      if self.factor_lengths is not None and self.factor_lengths[q] < self.min_factor_length:
        continue

      # If this state has inverse failures, its pointer will come up again, so don't return it.
      pointers = self.dfa.weights[q]
      # if pointers is not None:
      counter += len(pointers)
      for ptr in pointers:
        if self.retrieved[ptr] != self.n_retrievals:
          self.retrieved[ptr] = self.n_retrievals
          yield ptr

      queue.extend(self.inverse_failures[q])
