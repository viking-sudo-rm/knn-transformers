import torch

from .wfa import WFA
from .semiring import PointerSemiring


class TrieBuilder:

  """Build a linear chain DFA representing a dataset.
  
  This is a special case of a trie where the formal language to be represented is one string.

  General tries are currently not supported, as the labels are indices into the single input string.
  """

  def __init__(self):
    self.dfa = WFA(PointerSemiring())
    self.initial = self.dfa.new_state([0])
    self.solid_states = [self.initial]

  def build(self, tokens):
    """Augment the WFA with a trie path giving weight to tokens.

    Args:
      tokens: Path of tokens to add.
    """
    last_state = self.initial
    for idx, token in enumerate(tokens):
      if isinstance(token, torch.Tensor):
        token = token.item()
      state, _ = self.dfa.next_state(last_state, token)
      if state is None:
        state = self.dfa.new_state([idx + 1])
        self.solid_states.append(state)
        self.dfa.add_edge(last_state, token, state, None)
      last_state = state
