from unittest import TestCase

from src.semiring import PointerSemiring
from src.wfa import WFA
from src import serialize_dfa


# FIXME: Don't support serializing string DFAs!
# string_dfa = WFA(BooleanSemiring())
# string_dfa.new_state()
# string_dfa.new_state(weight=True)
# string_dfa.add_edge(0, "a", 1)
# string_dfa.add_edge(0, "b", 0)
# string_dfa.add_edge(1, "b", 0)


dfa = WFA(PointerSemiring())
dfa.new_state([2])
dfa.new_state([3, 4])
dfa.add_edge(0, 0, 1)
dfa.add_edge(0, 1, 0)
dfa.add_edge(1, 1, 0)


class SerializeDfaTest(TestCase):

    def test_serialize_transitions(self):
        flat_transitions, lengths = serialize_dfa.get_flat_transitions(dfa)
        flat_transitions = list(flat_transitions)
        lengths = list(lengths)
        self.assertListEqual(flat_transitions, [0, 1, 1, 0, 1, 0])
        self.assertListEqual(lengths, [4, 2])
        transitions = serialize_dfa.get_transitions(flat_transitions, lengths)
        self.assertEqual(transitions, dfa.transitions)

    def test_serialize_weights(self):
        flat_weights, lengths = serialize_dfa.get_flat_weights(dfa)
        flat_weights = list(flat_weights)
        lengths = list(lengths)
        self.assertListEqual(flat_weights, [2, 3, 4])
        self.assertListEqual(lengths, [1, 2])
