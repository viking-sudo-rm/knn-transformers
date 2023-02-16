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

  def get_initial_states(self):
    return torch.LongTensor([self.dfa.initial]) if self.add_initial else torch.LongTensor([])

  def get_pointers(self, states) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a list of pointers, as well as the states they came from."""
    pointers = []
    paired_states = []
    # FIXME: Tensor iterator memory leak.
    for idx in range(len(states)):
      state = states[idx]
      # For some reason, state.item() is critical here?
      pointers_gen = self.gen_pointers_from_state(state.item())
      if self.solid_only:
        pointers_gen = (ptr for ptr in pointers_gen if self.solid_states[ptr] == state)
      if self.max_pointers != -1:
        pointers_gen = islice(pointers_gen, self.max_pointers)
      prev_len = len(pointers)
      pointers.extend(pointers_gen)
      new_len = len(pointers)
      paired_states.extend(state for _ in range(new_len - prev_len))
    # Long tensor required for using pointers as indices.
    return torch.LongTensor(pointers), torch.LongTensor(paired_states)

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
        yield ptr

      if q in self.inverse_failures:
        queue.extend(self.inverse_failures[q])

  @torch.no_grad()
  def transition(self, states, token, neg_dists=None):
    """Follow state transitions given token.

    If there are too many valid states after transitioning, filter them based on distance.
    
    FIXME: Use pre-allocated arrays instead of lists."""
    indices = []
    next_states = []
    for idx in range(len(states)):
      state = states[idx].item()
      next_state = self.dfa.next_state(state, token)
      if not self.filter_state(next_state):
        indices.append(idx)
        next_states.append(next_state)

    next_states = torch.LongTensor(next_states)
    indices = torch.LongTensor(indices)
    if neg_dists is None or self.max_states == -1:
      return next_states

    neg_dists = neg_dists[indices]
    perm = neg_dists.argsort(descending=True)
    next_states = next_states[perm][:self.max_states]
    indices = indices[perm][:self.max_states]
    return next_states, indices

  @torch.no_grad()
  def solidify(self, states, knns):
    """Look up the solid state for pointers coming from KNN search."""
    knns = knns.numpy()
    solid_states = torch.tensor(self.solid_states[knns])
    return torch.where(states == -1, solid_states, states)

  def filter_state(self, state: int) -> bool:
    return state == -1 or (self.factor_lengths is not None and self.factor_lengths[state] < self.min_factor_length)
    # return state is None or (self.factor_lengths is not None and self.factor_lengths[state] < self.min_factor_length)