import unittest

from src.suffix_dfa_builder import SuffixDfaBuilder
from src.retriever import Retriever
from src.retriever_builder import RetrieverBuilder
from src.wfa import WFA
from src.semiring import PointerSemiring


class RetrieverTest(unittest.TestCase):

  def test_retrieve_ab_from_abcab(self):
    """ab is a factor of abcab. Should retrieve [2, 5]."""
    dfa = SuffixDfaBuilder().build("abcab")
    retriever = RetrieverBuilder().build(dfa)
    state = dfa.transition("ab")
    pointers = [ptr for ptr, _ in retriever.gen_pointers([state])]
    self.assertEqual(pointers, [2, 5])

  def test_retrieve_da_from_abcab(self):
    """Should retrieve indices of a, and requires handling unknown symbol correctly."""
    dfa = SuffixDfaBuilder().build("abcab")
    retriever = RetrieverBuilder().build(dfa)
    state = dfa.transition("da")
    pointers = [ptr for ptr, _ in retriever.gen_pointers([state])]
    self.assertEqual(pointers, [1, 4])

  def test_retrieve_cb_from_abcab(self):
    """Should retrive indices of b."""
    dfa = SuffixDfaBuilder().build("abcab")
    retriever = RetrieverBuilder().build(dfa)
    state = dfa.transition("cb")
    pointers = [ptr for ptr, _ in retriever.gen_pointers([state])]
    self.assertEqual(pointers, [2, 5])

  def test_retrieve_empty_string(self):
    """When the only thing that matches is the empty string, we should retrieve everything, and avoid an infinite loop."""
    dfa = SuffixDfaBuilder().build("abcab")
    retriever = RetrieverBuilder().build(dfa)
    state = dfa.transition("d")
    pointers = [ptr for ptr, _ in retriever.gen_pointers([state])]
    result = list(range(0, 6))
    self.assertEqual(pointers, result)

  def test_retrieve_long_unique(self):
    """Should return two different pointers to occurences of abab."""
    # Longer string with same problem:
    # string = "abcsdfsfdabcabcdababa"
    string = "cababa"
    factor = "ab"
    dfa = SuffixDfaBuilder().build(string)
    state = dfa.transition(factor)
    retriever = RetrieverBuilder().build(dfa)
    pointers = [ptr for ptr, _ in retriever.gen_pointers([state])]
    strings = [string[:ptr] for ptr in pointers]
    self.assertListEqual(strings, ["cab", "cabab"])

  def test_iterated_growth(self):
    string = "dcbacbabaa"
    factor = "ba"
    dfa = SuffixDfaBuilder().build(string)
    state = dfa.transition(factor)
    retriever = RetrieverBuilder().build(dfa)
    pointers = [ptr for ptr, _ in retriever.gen_pointers([state])]
    strings = [string[:ptr] for ptr in pointers]
    self.assertListEqual(strings, ["dcba", "dcbacba", "dcbacbaba"])

  def test_parallel(self):
    dfa = WFA(PointerSemiring())
    dfa.new_state([1, 2])
    retriever = RetrieverBuilder().build(dfa)
    pointers = [ptr for ptr, _ in retriever.gen_pointers([0, 0])]
    self.assertListEqual(pointers, [1, 2, 1, 2])
