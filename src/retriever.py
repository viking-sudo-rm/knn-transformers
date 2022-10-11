from collections import defaultdict
from typing import Iterable


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
          yield ptr, q

      if q in self.inverse_failures:
        queue.extend(self.inverse_failures[q])
