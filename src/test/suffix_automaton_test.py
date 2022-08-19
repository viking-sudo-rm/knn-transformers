import unittest
import numpy as np

from src.suffix_automaton import SuffixAutomatonBuilder


class SuffixAutomatonTest(unittest.TestCase):

  def test_abb(self):
    # Should return a 7-state automaton with two paths: ab^2 and b^2.
    string = "abb"
    builder = SuffixAutomatonBuilder()
    builder.build(string)
    builder.add_failures()
    self.assertDictEqual(builder.dfa.weights, {0: [-1, 0, 1, 2], 1: [0], 2: [1], 3: [2], 4: [1, 2]})
    self.assertDictEqual(builder.dfa.transitions,
                         {(0, "a"): (1, None),
                          (1, "b"): (2, None),
                          (2, "b"): (3, None),
                          (0, "b"): (4, None),
                          (4, "b"): (3, None)})
    self.assertEqual(builder.dfa.forward("abb"), [2])
    self.assertEqual(builder.dfa.forward("ab"), [1])

  def test_abcab(self):
    string = "abcab"
    builder = SuffixAutomatonBuilder()
    builder.build(string)
    builder.add_failures()
    self.assertEqual(builder.dfa.forward("ca"), [3])

  def test_build_on_int_array(self):
    """In practice we will pass in an array of ints."""
    string = np.array([0, 1, 2])
    builder = SuffixAutomatonBuilder()
    builder.build(string)
    builder.add_failures()
    dfa = builder.dfa
    result12 = dfa.forward(np.array([1, 2]))
    self.assertEqual(result12, [2])

  def test_build_long(self):
    string = "cababa"
    builder = SuffixAutomatonBuilder()
    builder.build(string)
    builder.add_failures()
    self.assertListEqual(builder.dfa.forward("ab"), [2, 4])
