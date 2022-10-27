import unittest
import torch

from src.trie_builder import TrieBuilder


class TrieBuilderTest(unittest.TestCase):

    def test_build_string(self):
        builder = TrieBuilder()
        builder.build("abab")
        dfa = builder.dfa
        self.assertListEqual(
            list(dfa.weights.values()), [[0], [1], [2], [3], [4]])
        self.assertDictEqual(dfa.transitions,
        {
            (0, "a"): (1, None),
            (1, "b"): (2, None),
            (2, "a"): (3, None),
            (3, "b"): (4, None),
        }
        )

        self.assertListEqual(builder.solid_states, [0, 1, 2, 3, 4])

    def test_build_tensor(self):
        builder = TrieBuilder()
        builder.build(torch.tensor([1, 1, 1]))
        dfa = builder.dfa
        weights = list(dfa.weights.values())
        self.assertIsInstance(weights[0][0], int)
        for _, token in dfa.transitions.keys():
            self.assertIsInstance(token, int)
        self.assertListEqual(weights, [[0], [1], [2], [3]])
        self.assertDictEqual(dfa.transitions, {
            (0, 1): (1, None),
            (1, 1): (2, None),
            (2, 1): (3, None),
        })
