import unittest
import numpy as np
import torch

from src.suffix_dfa_builder import SuffixDfaBuilder


class SuffixDfaBuilderTest(unittest.TestCase):


  def test_abb_behavior(self):
    string = "abb"
    builder = SuffixDfaBuilder(len(string))
    builder.build(string)
    self.assertEqual(builder.dfa.forward("abb"), 3)
    self.assertEqual(builder.dfa.forward("ab"), 2)
    self.assertEqual(builder.dfa.forward("aa"), 1)

  def test_abb_internals(self):
    string = "abb"
    builder = SuffixDfaBuilder(len(string))
    builder.build(string)
    self.assertEqual(builder.dfa.transitions,
                     [[('a', 1), ('b', 4)], [('b', 2)], [('b', 3)], [], [('b', 3)]])
    self.assertListEqual(builder.dfa.weights.tolist(),
                         [0,  1,  2,  3, -1, -1])
    # FIXME: The suffix DFA has a self-loop, which is a problem.

  def test_abcab(self):
    string = "abcab"
    builder = SuffixDfaBuilder(len(string))
    builder.build(string)
    self.assertEqual(builder.dfa.forward("ca"), [4])

  def test_build_on_int_array(self):
    """In practice we will pass in an array of ints."""
    string = np.array([0, 1, 2])
    builder = SuffixDfaBuilder(len(string))
    builder.build(string)
    dfa = builder.dfa
    result12 = dfa.forward(np.array([1, 2]))
    self.assertEqual(result12, [3])

  def test_build_long(self):
    string = "cababa"
    builder = SuffixDfaBuilder(len(string))
    builder.build(string)
    self.assertEqual(builder.dfa.transition("ab"), 7)
    self.assertEqual(builder.dfa.forward("ab"), -1)

    # Check that states correspond to positions correctly.
    self.assertListEqual(builder.dfa.weights.tolist(),
                         [0,  1,  2,  3,  4, -1,  5, -1,  6, -1, -1, -1])
    # Check that the state we found retrieves the right positions.
    self.assertEqual(builder.dfa.failures[3], 7)
    self.assertEqual(builder.dfa.failures[6], 7)

  def test_build_on_tensor(self):
    """In practice we will pass in an array of ints."""
    string = torch.tensor([0, 1, 2])
    builder = SuffixDfaBuilder(len(string))
    builder.build(string)
    self.assertEqual(builder.dfa.weights.dtype, np.int32)
    self.assertEqual(builder.dfa.weights.tolist(),
                     [ 0,  1,  2,  3, -1, -1])
    for token, _ in builder.dfa.transitions[0]:
      self.assertIsInstance(token, np.int32)

  def test_build_oracle(self):
    string = "abbbaab"
    builder = SuffixDfaBuilder(len(string), oracle=True)
    dfa = builder.build(string)
    breakpoint()