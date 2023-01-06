import pywrapfst as fst

from .wfa import WFA


def convert_to_pywrapfst(dfa: WFA) -> fst.VectorFst:
  """See here: https://www.openfst.org/twiki/bin/view/FST/PythonExtension"""
  f = fst.VectorFst()
  f.reserve_states(dfa.n_states)
  one = fst.Weight.one(f.weight_type())
  # FIXME: Change weight type/weights here. Meaningless right now.

  # Create all the states.
  for weight in dfa.weights:
    if weight != -1:
      q = f.add_state()
      f.set_final(q, fst.Weight(f.weight_type(), weight))

  # Create all the edges.
  for state, state_trans in enumerate(dfa.transitions):
    for token, next_state in state_trans:
      arc = fst.Arc(token, token, one, next_state)
      f.add_arc(state, arc)

  # Create failure transitions.
  if dfa.failures is not None:
    for state, phi in enumerate(dfa.failures):
      if phi == -1:
        continue
      arc = fst.Arc(-1, -1, one, phi)
      f.add_arc(state, arc)

  # Set the initial state.
  f.set_start(0)

  return f