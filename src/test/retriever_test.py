import unittest
import torch

from src.suffix_dfa_builder import SuffixDfaBuilder
from src.trie_builder import TrieBuilder
from src.retriever import Retriever
from src.retriever_builder import RetrieverBuilder
from src.wfa import WFA
from src.semiring import PointerSemiring


dfa = WFA(4, failures=True)
dfa.add_state(weight=0)
dfa.add_state(weight=1)
dfa.add_state(weight=2)
dfa.add_state(weight=2)
dfa.add_edge(0, "a", 1)
dfa.add_edge(1, "a", 2)
dfa.add_edge(2, "b", 3)
dfa.failures = {0: 0, 1: 0, 2: 0}
inverse_failures = {0: [1, 2]}
dfa.solid_states = [0, 1, 2]
abcab: WFA = SuffixDfaBuilder(5).build("abcab")


class RetrieverTest(unittest.TestCase):

  def test_retrieve_ab_from_abcab(self):
    """ab is a factor of abcab. Should retrieve [2, 5]."""
    retriever = RetrieverBuilder().build(abcab)
    state = abcab.transition("ab")
    pointers = retriever.get_pointers(torch.LongTensor([state]))
    self.assertEqual(pointers.tolist(), [2, 5])

  def test_retrieve_da_from_abcab(self):
    """Should retrieve indices of a, and requires handling unknown symbol correctly."""
    retriever = RetrieverBuilder().build(abcab)
    state = abcab.transition("da")
    pointers = retriever.get_pointers(torch.LongTensor([state]))
    self.assertEqual(pointers.tolist(), [1, 4])

  def test_retrieve_cb_from_abcab(self):
    """Should retrive indices of b."""
    retriever = RetrieverBuilder().build(abcab)
    state = abcab.transition("cb")
    pointers = retriever.get_pointers(torch.LongTensor([state]))
    self.assertEqual(pointers.tolist(), [2, 5])

  def test_retrieve_empty_string(self):
    """When the only thing that matches is the empty string, we should retrieve everything, and avoid an infinite loop."""
    retriever = RetrieverBuilder().build(abcab)
    state = abcab.transition("d")
    states = torch.LongTensor(torch.LongTensor([state]))
    pointers = retriever.get_pointers(states)
    result = list(range(0, 6))
    self.assertEqual(pointers.tolist(), result)

  def test_retrieve_long_unique(self):
    """Should return two different pointers to occurences of abab."""
    # Longer string with same problem:
    # string = "abcsdfsfdabcabcdababa"
    string = "cababa"
    factor = "ab"
    dfa = SuffixDfaBuilder(len(string)).build(string)
    state = dfa.transition(factor)
    retriever = RetrieverBuilder().build(dfa)
    states = torch.LongTensor(torch.LongTensor([state]))
    pointers = retriever.get_pointers(states)
    strings = [string[:ptr] for ptr in pointers]
    self.assertListEqual(strings, ["cab", "cabab"])

  def test_iterated_growth(self):
    string = "dcbacbabaa"
    factor = "ba"
    dfa = SuffixDfaBuilder(len(string)).build(string)
    state = dfa.transition(factor)
    retriever = RetrieverBuilder().build(dfa)
    states = torch.LongTensor(torch.LongTensor([state]))
    pointers = retriever.get_pointers(states)
    strings = {string[:ptr] for ptr in pointers}
    self.assertSetEqual(strings, {"dcba", "dcbacba", "dcbacbaba"})

  def test_abb_no_fail_first(self):
    string = "abb"
    dfa = SuffixDfaBuilder(len(string)).build(string)
    builder = RetrieverBuilder(min_factor_length=1)
    retriever = builder.build(dfa)
    pointers = retriever.get_pointers(torch.LongTensor([3]))
    # Don't fail back to the empty string state, but do keep length 1 state.
    self.assertListEqual(pointers.tolist(), [3])

  def test_abb_fail_first(self):
    string = "abb"
    dfa = SuffixDfaBuilder(len(string)).build(string)
    builder = RetrieverBuilder(min_factor_length=1, fail_first=True)
    retriever = builder.build(dfa)
    pointers = retriever.get_pointers(torch.LongTensor([3]))
    # Don't fail back to the empty string state, but do keep length 1 state.
    self.assertListEqual(pointers.tolist(), [2, 3])

  def test_transition_basic(self):
    retriever = Retriever(dfa, {}, None)
    pointers = torch.tensor([0, 1])
    states = retriever.get_states(pointers, "a")
    self.assertListEqual(states.tolist(), [1, 2])

  def test_transition_invalid(self):
    string = "ab"
    dfa = TrieBuilder(len(string)).build(string)
    retriever = Retriever(dfa, {}, None)
    pointers = torch.tensor([0, 1])
    states = retriever.get_states(pointers, "b")
    self.assertListEqual(states.tolist(), [2])
  
  def test_transition_all_invalid(self):
    """Should throw out all states."""
    string = "aa"
    dfa = TrieBuilder(len(string)).build(string)
    retriever = Retriever(dfa, {}, None)
    states = retriever.get_states(torch.tensor([0, 1]), "b")
    self.assertListEqual(states.tolist(), [])

  def test_get_pointers_solid_only(self):
    retriever = Retriever(dfa, {}, None, solid_only=True)
    states = torch.LongTensor([3])
    pointers = retriever.get_pointers(states)
    self.assertListEqual(pointers.tolist(), [])

  def test_get_states_cutoff(self):
    retriever = Retriever(dfa, {}, None, max_states=1)
    pointers = torch.tensor([0, 1, 2])
    dists = torch.tensor([1., .5, 0.])
    states = retriever.get_states(pointers, "a", dists=dists)
    # Follow the transition from 0 -> 1.
    self.assertListEqual(states.tolist(), [1])

  def test_gen_pointers_from_state(self):
    retriever = Retriever(dfa, inverse_failures, None)
    pointers = [ptr for ptr, _ in retriever.gen_pointers_from_state(0)]
    self.assertListEqual(pointers, [0, 1, 2])

  def test_gen_pointers_from_state_cutoff(self):
    retriever = Retriever(dfa, inverse_failures, None, max_pointers=2)
    pointers = [ptr for ptr, _ in retriever.gen_pointers_from_state(0)]
    self.assertListEqual(pointers, [0, 1])

  def test_get_pointers(self):
    retriever = Retriever(dfa, inverse_failures, None)
    states = torch.tensor([0])
    pointers = retriever.get_pointers(states)
    self.assertListEqual(pointers.tolist(), [0, 1, 2])

  def test_get_pointers_cutoff(self):
    retriever = Retriever(dfa, inverse_failures, None, max_pointers=2)
    states = torch.tensor([0])
    pointers = retriever.get_pointers(states)
    self.assertListEqual(pointers.tolist(), [0, 1])
