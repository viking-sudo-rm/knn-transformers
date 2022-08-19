from collections import defaultdict
from typing import Iterable


class Retriever:

  """Retrieve contexts that are similar to the context."""

  def __init__(self, dfa, max_factor_length: int = 768, max_pointers: int = 1024):
    self.dfa = dfa
    self.max_factor_length = max_factor_length
    self.max_pointers = max_pointers
    self.inverse_failures = self._get_inverse_failures()
    self.n_retrievals = 0
    self.retrieved = defaultdict(int)

  def _get_inverse_failures(self):
    inverse_failures = defaultdict(list)
    for state, fail_state in self.dfa.failures.items():
      if state is None or fail_state is None:
        continue
      inverse_failures[fail_state].append(state)
    inverse_failures[self.dfa.initial].remove(self.dfa.initial)
    return inverse_failures

  def gen_pointers(self, context: str) -> Iterable[int]:
    """Get pointers to occurences of the longest factor suffix of context."""
      # Track which retrieval we are in.
    self.n_retrievals += 1
    state, _ = self.dfa.transition(context)
    # state, _ = self._get_largest_factor_suffix(context)
    if state is None:
      return

    # TODO: Also follow chain of failures from state?
    # while state != self.dfa.initial:
    queue = [state]
    counter = 0
    while counter < self.max_pointers:
      if not queue:
        break
      q = queue.pop(0)

      # If this state has inverse failures, its pointer will come up again, so don't return it.
      pointers = self.dfa.weights[q]
      if pointers is not None:
        counter += len(pointers)
        for ptr in pointers:
          if self.retrieved[ptr] != self.n_retrievals:
            self.retrieved[ptr] = self.n_retrievals
            yield ptr

      # FIXME(lambdaviking): Do we need to worry about the case where something fails to longer string but not shorter one?
      inverse_failures = self.inverse_failures[q]
      # queue.extend(q_ for q_ in inverse_failures if self.dfa.weights[q_] != pointers)
      queue.extend(inverse_failures)
    # state = self.dfa.failures[state]
