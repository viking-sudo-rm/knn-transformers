from unittest import TestCase
import shutil
import tempfile

from src.wfa import WFA
from src import serialize_dfa


dfa = WFA(2)
dfa.add_state(10)
dfa.add_state(11)
dfa.add_edge(0, 0, 1)
dfa.add_edge(0, 1, 0)
dfa.add_edge(1, 1, 0)

fail_dfa = WFA(1, failures=True)
fail_dfa.add_state(10)
fail_dfa.failures[0] = 0


class SerializeDfaTest(TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_serialize_transitions(self):
        flat_transitions, lengths = serialize_dfa.get_flat_transitions(dfa)
        flat_transitions = list(flat_transitions)
        lengths = list(lengths)
        self.assertListEqual(flat_transitions, [0, 1, 1, 0, 1, 0])
        self.assertListEqual(lengths, [4, 2])
        transitions = serialize_dfa.get_transitions(flat_transitions, lengths)
        self.assertEqual(transitions, dfa.transitions)

    def test_save_load(self):
        serialize_dfa.save_dfa(self.test_dir, dfa)
        dfa2 = serialize_dfa.load_dfa(self.test_dir)
        self.assertListEqual(dfa2.weights.tolist(), [10, 11])
        self.assertEqual(dfa2.transitions, [[(0, 1), (1, 0)], [(1, 0)]])
        self.assertIsNone(dfa2.failures)
        self.assertEqual(dfa2.n_states, 2)
        self.assertEqual(dfa2.initial, 0)

    def test_save_load_failures(self):
        serialize_dfa.save_dfa(self.test_dir, fail_dfa)
        dfa2 = serialize_dfa.load_dfa(self.test_dir)
        self.assertListEqual(dfa2.weights.tolist(), [10])
        self.assertEqual(dfa2.transitions, [[]])
        self.assertEqual(dfa2.failures.tolist(), [0])
        self.assertEqual(dfa2.n_states, 1)
        self.assertEqual(dfa2.initial, 0)
