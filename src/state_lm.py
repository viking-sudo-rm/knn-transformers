import torch

from .wfa import WFA


@torch.no_grad()
def get_transitions(dfa: WFA, vocab_size, device="cpu"):
  indices = -torch.ones((2, 2 * dfa.n_states + 1), dtype=torch.int32)
  values = torch.zeros((2 * dfa.n_states + 1,), dtype=torch.bool)
  degrees = torch.zeros(dfa.n_states, dtype=torch.int32)
  used_failures = dfa._failures
  dfa.use_failures(False)
  counter = 0
  for state in range(dfa.n_states):
    if state >= len(dfa.transitions):
      continue
    # Tensor iterator memory leak
    trans = dfa.transitions[state]
    # for token, _ in dfa.transitions[state]:
    for idx in range(len(trans)):
      token, _ = trans[idx]
      indices[0, counter] = state
      indices[1, counter] = token
      values[counter] = True
      degrees[state] += 1
      counter += 1
  dfa.use_failures(used_failures)
  mask = (indices[0] != -1)
  indices = indices[:, mask]
  values = values[mask]
  size = (dfa.n_states, vocab_size)
  transitions = torch.sparse_coo_tensor(indices, values, size)
  return transitions, degrees


@torch.no_grad()
def get_transitions_subset(states, dfa, n_vocab, device="cpu"):
  transitions = torch.zeros(len(states), n_vocab, dtype=torch.bool, device=device)
  for i in range(len(states)):
    state = states[i]
    trans = dfa.transitions[state]
    for j in range(len(trans)):
      token, _ = trans[j]
      transitions[i, token] = True
  return transitions


class StateLm:

  """Nonparametric language model based on active states and their distances.
  
  If self.transitions is None, should specify vocab size."""

  def __init__(self,
               dfa: WFA,
               vocab_size: int,
               temp: float = 1.,
               mode: str = "uniform-nofail",
               transitions=None,
               degrees=None,
               device="cpu",
              ):
    """Create a state LM given some transition weights."""
    self.dfa = dfa
    self.vocab_size = vocab_size
    self.temp = temp
    self.mode = mode
    self.transitions = transitions
    self.degrees = degrees
    self.device = device

  @classmethod
  def create_memoized(cls, dfa, vocab_size, device="cpu", **kwargs):
    transitions, degrees = get_transitions(dfa, vocab_size, device=device)
    return cls(dfa, vocab_size, **kwargs, transitions=transitions, degrees=degrees, device=device)

  def get_log_prob(self, states, neg_dists):
    """Follows knns_to_log_prob in knnlm.py."""
    if self.transitions is None:
      transitions = get_transitions_subset(states, self.dfa, self.vocab_size, device=self.device)
      degrees = (transitions > 0).int().sum(dim=-1)
    else:
      transitions = torch.index_select(self.transitions, 0, states).to_dense()  # (k, n_vocab)
      degrees = torch.index_select(self.degrees, 0, states)  # (k,)

    if self.mode == "uniform-nofail":
      p_token_given_ptr = transitions / degrees.unsqueeze(-1)  # (k, n_vocab)
      # NaN should only come up if we receive a state with no out transitions, in which case probability mass is missing (belongs to EOS)
      p_token_given_ptr = torch.where(p_token_given_ptr.isnan(), torch.zeros_like(p_token_given_ptr), p_token_given_ptr)
    else:
      raise NotImplementedError
    p_ptr = torch.nn.functional.softmax(neg_dists / self.temp, dim=-1)  # (k,)
    log_probs = (p_ptr @ p_token_given_ptr).log()
    log_probs = torch.nan_to_num(log_probs, nan=None, neginf=-10000.0)
    return log_probs
