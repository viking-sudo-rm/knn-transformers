import unittest

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