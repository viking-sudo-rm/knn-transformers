import unittest
import torch
import numpy as np

from src.trie_builder import TrieBuilder


class TrieBuilderTest(unittest.TestCase):

    def test_build_string(self):
        builder = TrieBuilder()
        builder.build("abab")
        dfa = builder.dfa
        self.assertListEqual(dfa.weights,
                             [[0], [1], [2], [3], [4]])
        self.assertEqual(dfa.transitions,
                         [[("a", 1)], [("b", 2)], [("a", 3)], [("b", 4)], []])
        self.assertListEqual(builder.solid_states, [0, 1, 2, 3, 4])

    def test_build_tensor(self):
        builder = TrieBuilder()
        builder.build(torch.tensor([1, 1, 1]))
        dfa = builder.dfa
        self.assertIsInstance(dfa.weights[0][0], np.int32)
        for token, _ in dfa.transitions[0]:
            self.assertIsInstance(token, np.int32)
        self.assertListEqual(dfa.weights,
                             [[0], [1], [2], [3]])
        self.assertEqual(dfa.transitions,
                         [[(1, 1)], [(1, 2)], [(1, 3)], []])
