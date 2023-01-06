import torch
import numpy as np

from .wfa import WFA
from .semiring import PointerSemiring
from .type_utils import to_int32


class TrieBuilder:

  """Build a linear chain DFA representing a dataset.
  
  This is a special case of a trie where the formal language to be represented is one string.

  General tries are currently not supported, as the labels are indices into the single input string.
  """

  def __init__(self, dstore_size):
    n_states = dstore_size + 1
    self.dfa = WFA(n_states)
    self.initial = self.dfa.add_state(0)
    self.solid_states = -np.ones(n_states, dtype=np.int32)

  def build(self, tokens):
    """Augment the WFA with a trie path giving weight to tokens.

    Args:
      tokens: Path of tokens to add.
    """
    last_state = self.initial
    self.solid_states[0] = 0
    # for idx, token in enumerate(tokens):
    for idx in range(len(tokens)):
      token = tokens[idx]
      token = to_int32(token)
      idx = to_int32(idx)
      state = self.dfa.next_state(last_state, token)
      if state == -1:
        state = self.dfa.add_state(idx + 1)
        # self.solid_states.append(state)
        self.solid_states[idx + 1] = state
        self.dfa.add_edge(last_state, token, state)
      last_state = state
    self.dfa.solid_states = self.solid_states
    return self.dfa
