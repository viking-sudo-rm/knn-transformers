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
    builder.add_failures()
    self.assertDictEqual(builder.dfa.weights, {0: [0, 1, 2, 3], 1: [1], 2: [2], 3: [3], 4: [2, 3]})
    self.assertDictEqual(builder.dfa.transitions,
                         {(0, "a"): (1, None),
                          (1, "b"): (2, None),
                          (2, "b"): (3, None),
                          (0, "b"): (4, None),
                          (4, "b"): (3, None)})
    self.assertEqual(builder.dfa.forward("abb"), [3])
    self.assertEqual(builder.dfa.forward("ab"), [2])

  def test_abcab(self):
    string = "abcab"
    builder = SuffixDfaBuilder()
    builder.build(string)
    builder.add_failures()
    self.assertEqual(builder.dfa.forward("ca"), [4])

  def test_build_on_int_array(self):
    """In practice we will pass in an array of ints."""
    string = np.array([0, 1, 2])
    builder = SuffixDfaBuilder()
    builder.build(string)
    builder.add_failures()
    dfa = builder.dfa
    result12 = dfa.forward(np.array([1, 2]))
    self.assertEqual(result12, [3])

  def test_build_long(self):
    string = "cababa"
    builder = SuffixDfaBuilder()
    builder.build(string)
    builder.add_failures()
    self.assertListEqual(builder.dfa.forward("ab"), [3, 5])

  def test_build_on_tensor(self):
    """In practice we will pass in an array of ints."""
    string = torch.tensor([0, 1, 2])
    builder = SuffixDfaBuilder()
    builder.build(string)
    for _, token in builder.dfa.transitions.keys():
      self.assertIsInstance(token, int)
