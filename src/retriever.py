from collections import defaultdict
from itertools import islice
from typing import Iterable
import torch


class Retriever:

  """Retrieve contexts that are similar to the context."""

  def __init__(self,
               dfa,
               inverse_failures,
               factor_lengths,
               min_factor_length: int = 0,
               max_pointers: int = -1,
               max_states: int = -1,
               fail_first: bool = False,
               add_initial: bool = True,
               solid_only: bool = False,
              ):
    self.dfa = dfa
    self.solid_states = self.dfa.solid_states
    self.inverse_failures = inverse_failures
    self.factor_lengths = factor_lengths
    self.min_factor_length = min_factor_length
    self.max_pointers = max_pointers  # per state
    self.max_states = max_states
    self.fail_first = fail_first
    self.add_initial = add_initial
    self.solid_only = solid_only

    # self.n_retrievals = 0
    # self.retrieved = defaultdict(int)

  def get_initial_states(self):
    return torch.LongTensor([self.dfa.initial]) if self.add_initial else torch.LongTensor([])

  def get_pointers(self, states):
    pointers = []
    for state in states:
      # For some reason, state.item() is critical here?
      pointers_gen = self.gen_pointers_from_state(state.item())
      if self.solid_only:
        pointers_gen = ((ptr, x) for ptr, x in pointers_gen if self.solid_states[ptr] == state)
      if self.max_pointers != -1:
        pointers_gen = islice(pointers_gen, self.max_pointers)
      pointers.extend(ptr for ptr, _ in pointers_gen)
    # Long tensor required for using pointers as indices.
    return torch.LongTensor(pointers)

  @torch.no_grad()
  def get_states(self, pointers, token, dists=None):
    indices = []
    states = []
    # Enumerating tensors is gross for some reason.
    for idx in range(len(pointers)):
      ptr = pointers[idx].item()
      state = self.solid_states[ptr]
      next_state = self.dfa.next_state(state, token)
      if not self.filter_state(next_state):
        indices.append(idx)
        states.append(next_state)

    # FIXME: Use pre-allocated arrays instead of lists.

    states = torch.LongTensor(states)
    if dists is None:
      return states
    dists = dists[indices]
    states = states[dists.argsort(descending=True)]
    if self.max_states != -1:
        states = states[:self.max_states]
    return states

  def filter_state(self, state: int) -> bool:
    return state == -1 or (self.factor_lengths is not None and self.factor_lengths[state] < self.min_factor_length)
    # return state is None or (self.factor_lengths is not None and self.factor_lengths[state] < self.min_factor_length)

  def gen_pointers_from_state(self, state: int) -> Iterable[int]:
    """Get pointers out of a state in the DFA."""
    counter = 0

    # Essentially: retrieve all occurrences of suffix of length k in the dataset.
    if self.fail_first and self.factor_lengths is not None and self.dfa.failures is not None:
      while self.factor_lengths[state] > self.min_factor_length:
        if self.dfa.failures[state] == -1:
          break
        state = self.dfa.failures[state]

    queue = [state]
    while self.max_pointers == -1 or counter < self.max_pointers:
      if not queue:
        break
      q = queue.pop(0)
      if self.factor_lengths is not None and self.factor_lengths[q] < self.min_factor_length:
        continue

      ptr = self.dfa.weights[q]
      if ptr != -1:
        counter += 1
        yield ptr, q

      if q in self.inverse_failures:
        queue.extend(self.inverse_failures[q])
