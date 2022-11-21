import unittest
import numpy as np
import torch

from src.suffix_dfa_builder import SuffixDfaBuilder


class SuffixDfaBuilderTest(unittest.TestCase):

  def test_abb(self):
    # Should return a 7-state automaton with two paths: ab^2 and b^2.
    string = "abb"
    builder = SuffixDfaBuilder()
    builder.build(string)
    self.assertEqual(builder.dfa.weights, [[0], [1], [2], [3], [2, 3]])
    self.assertEqual(builder.dfa.transitions,
                     [[('a', 1), ('b', 4)], [('b', 2)], [('b', 3)], [], [('b', 3)]])
    self.assertEqual(builder.dfa.forward("abb"), [3])
    self.assertEqual(builder.dfa.forward("ab"), [2])

  def test_abcab(self):
    string = "abcab"
    builder = SuffixDfaBuilder()
    builder.build(string)
    self.assertEqual(builder.dfa.forward("ca"), [4])

  def test_build_on_int_array(self):
    """In practice we will pass in an array of ints."""
    string = np.array([0, 1, 2])
    builder = SuffixDfaBuilder()
    builder.build(string)
    dfa = builder.dfa
    result12 = dfa.forward(np.array([1, 2]))
    self.assertEqual(result12, [3])

  def test_build_long(self):
    string = "cababa"
    builder = SuffixDfaBuilder()
    builder.build(string)
    self.assertListEqual(builder.dfa.forward("ab"), [3, 5])

  def test_build_on_tensor(self):
    """In practice we will pass in an array of ints."""
    string = torch.tensor([0, 1, 2])
    builder = SuffixDfaBuilder()
    builder.build(string)
    weights = list(builder.dfa.weights)
    self.assertIsInstance(weights[0][0], np.int32)
    for token, _ in builder.dfa.transitions[0]:
      self.assertIsInstance(token, np.int32)
