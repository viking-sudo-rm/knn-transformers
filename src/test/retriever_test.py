import unittest

from src.suffix_automaton import SuffixAutomatonBuilder
from src.retriever import Retriever


class RetrieverTest(unittest.TestCase):

  # def test_get_largest_factor_suffix_aaaaa(self):
  #   builder = SuffixAutomatonBuilder().build("aaaaa").add_failures()
  #   retriever = Retriever(builder.dfa)
  #   state, length = retriever._get_largest_factor_suffix("bbaaa")
  #   self.assertEqual(state, builder.dfa.transition("aaa")[0])
  #   # self.assertEqual(length, 3)

  # def test_get_largest_factor_suffix_abcab(self):
  #   builder = SuffixAutomatonBuilder().build("abcab").add_failures()
  #   retriever = Retriever(builder.dfa)
  #   breakpoint()
  #   state, length = retriever._get_largest_factor_suffix("cb")
  #   self.assertEqual(state, builder.dfa.transition("b")[0])
  #   self.assertEqual(length, 1)

  def test_retrieve_ab_from_abcab(self):
    """ab is a factor of abcab. Should retrieve [1, 4]."""
    builder = SuffixAutomatonBuilder().build("abcab").add_failures()
    retriever = Retriever(builder.dfa)
    state, _ = builder.dfa.transition("ab")
    pointers = list(retriever.gen_pointers([state]))
    self.assertEqual(pointers, [1, 4])

  def test_retrieve_da_from_abcab(self):
    """Should retrieve indices of a, and requires handling unknown symbol correctly."""
    builder = SuffixAutomatonBuilder().build("abcab").add_failures()
    retriever = Retriever(builder.dfa)
    state, _ = builder.dfa.transition("da")
    pointers = list(retriever.gen_pointers([state]))
    self.assertEqual(pointers, [0, 3])

  def test_retrieve_cb_from_abcab(self):
    """Should retrive indices of b."""
    builder = SuffixAutomatonBuilder().build("abcab").add_failures()
    retriever = Retriever(builder.dfa)
    state, _ = builder.dfa.transition("cb")
    pointers = list(retriever.gen_pointers([state]))
    self.assertEqual(pointers, [1, 4])

  def test_retrieve_empty_string(self):
    """When the only thing that matches is the empty string, we should retrieve everything, and avoid an infinite loop."""
    builder = SuffixAutomatonBuilder().build("abcab").add_failures()
    retriever = Retriever(builder.dfa)
    state, _ = builder.dfa.transition("d")
    pointers = list(retriever.gen_pointers([state]))
    self.assertEqual(pointers, [-1, 0, 1, 2, 3, 4])

  def test_retrieve_long_unique(self):
    """Should return two different pointers to occurences of abab."""
    # Longer string with same problem:
    # string = "abcsdfsfdabcabcdababa"
    string = "cababa"
    factor = "ab"
    builder = SuffixAutomatonBuilder().build(string).add_failures()
    state, _ = builder.dfa.transition(factor)
    retriever = Retriever(builder.dfa)
    pointers = list(retriever.gen_pointers([state]))
    strings = [string[:ptr + 1] for ptr in pointers]
    self.assertListEqual(strings, ["cab", "cabab"])

  def test_iterated_growth(self):
    string = "dcbacbabaa"
    factor = "ba"
    builder = SuffixAutomatonBuilder().build(string).add_failures()
    state, _ = builder.dfa.transition(factor)
    retriever = Retriever(builder.dfa)
    pointers = list(retriever.gen_pointers([state]))
    strings = [string[:ptr + 1] for ptr in pointers]
    self.assertListEqual(strings, ["dcba", "dcbacba", "dcbacbaba"])

  def test_retrieve_bounded_number(self):
    # FIXME(lambdaviking): This test case seems to be broken.
    string = "a" * 10
    factor = "a"
    builder = SuffixAutomatonBuilder().build(string).add_failures()
    state, _ = builder.dfa.transition(factor)
    retriever = Retriever(builder.dfa, max_pointers=2)
    pointers = list(retriever.gen_pointers([state]))
    self.assertEqual(len(pointers), 2)