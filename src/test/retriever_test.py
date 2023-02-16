import unittest
import torch
import numpy as np

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
dfa.solid_states = np.array([0, 1, 2], dtype=np.int32)
abcab: WFA = SuffixDfaBuilder(5).build("abcab")


class RetrieverTest(unittest.TestCase):

  def test_retrieve_ab_from_abcab(self):
    """ab is a factor of abcab. Should retrieve [2, 5]."""
    retriever = RetrieverBuilder().build(abcab)
    state = abcab.transition("ab")
    pointers, paired_states = retriever.get_pointers(torch.LongTensor([state]))
    self.assertListEqual(pointers.tolist(), [2, 5])
    self.assertListEqual(paired_states.tolist(), [state, state])

  def test_retrieve_da_from_abcab(self):
    """Should retrieve indices of a, and requires handling unknown symbol correctly."""
    retriever = RetrieverBuilder().build(abcab)
    state = abcab.transition("da")
    pointers, paired_states = retriever.get_pointers(torch.LongTensor([state]))
    self.assertListEqual(pointers.tolist(), [1, 4])
    self.assertListEqual(paired_states.tolist(), [state, state])

  def test_retrieve_cb_from_abcab(self):
    """Should retrive indices of b."""
    retriever = RetrieverBuilder().build(abcab)
    state = abcab.transition("cb")
    pointers, paired_states = retriever.get_pointers(torch.LongTensor([state]))
    self.assertListEqual(pointers.tolist(), [2, 5])
    self.assertListEqual(paired_states.tolist(), [state, state])

  def test_retrieve_empty_string(self):
    """When the only thing that matches is the empty string, we should retrieve everything, and avoid an infinite loop."""
    retriever = RetrieverBuilder().build(abcab)
    state = abcab.transition("d")
    states = torch.LongTensor(torch.LongTensor([state]))
    pointers, paired_states = retriever.get_pointers(states)
    result = list(range(6))
    self.assertListEqual(pointers.tolist(), result)
    self.assertListEqual(paired_states.tolist(), [state for _ in range(6)])

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
    pointers, paired_states = retriever.get_pointers(states)
    strings = [string[:ptr] for ptr in pointers]
    self.assertListEqual(strings, ["cab", "cabab"])
    self.assertListEqual(paired_states.tolist(), [state, state])

  def test_iterated_growth(self):
    string = "dcbacbabaa"
    factor = "ba"
    dfa = SuffixDfaBuilder(len(string)).build(string)
    state = dfa.transition(factor)
    retriever = RetrieverBuilder().build(dfa)
    states = torch.LongTensor(torch.LongTensor([state]))
    pointers, paired_states = retriever.get_pointers(states)
    strings = {string[:ptr] for ptr in pointers}
    self.assertSetEqual(strings, {"dcba", "dcbacba", "dcbacbaba"})
    self.assertListEqual(paired_states.tolist(), [state, state, state])

  def test_abb_no_fail_first(self):
    string = "abb"
    dfa = SuffixDfaBuilder(len(string)).build(string)
    builder = RetrieverBuilder(min_factor_length=1)
    retriever = builder.build(dfa)
    pointers, paired_states = retriever.get_pointers(torch.LongTensor([3]))
    # Don't fail back to the empty string state, but do keep length 1 state.
    self.assertListEqual(pointers.tolist(), [3])
    self.assertListEqual(paired_states.tolist(), [3])

  def test_abb_fail_first(self):
    string = "abb"
    dfa = SuffixDfaBuilder(len(string)).build(string)
    builder = RetrieverBuilder(min_factor_length=1, fail_first=True)
    retriever = builder.build(dfa)
    pointers, paired_states = retriever.get_pointers(torch.LongTensor([3]))
    # Don't fail back to the empty string state, but do keep length 1 state.
    self.assertListEqual(pointers.tolist(), [2, 3])
    self.assertListEqual(paired_states.tolist(), [3, 3])

  def test_transition_basic(self):
    retriever = Retriever(dfa, {}, None)
    states = torch.tensor([0, 1])
    states = retriever.transition(states, "a")
    self.assertListEqual(states.tolist(), [1, 2])

  def test_transition_invalid(self):
    string = "ab"
    dfa = TrieBuilder(len(string)).build(string)
    retriever = Retriever(dfa, {}, None)
    states = torch.tensor([0, 1])
    states = retriever.transition(states, "b")
    self.assertListEqual(states.tolist(), [2])

  def test_transition_all_invalid(self):
    """Should throw out all states."""
    string = "aa"
    dfa = TrieBuilder(len(string)).build(string)
    retriever = Retriever(dfa, {}, None)
    states = retriever.transition(torch.tensor([0, 1]), "b")
    self.assertListEqual(states.tolist(), [])

  def test_get_states_cutoff(self):
    dfa = WFA(3)
    dfa.add_state(weight=0)
    dfa.add_state(weight=1)
    dfa.add_state(weight=2)
    dfa.add_edge(0, "a", 1)
    dfa.add_edge(1, "a", 2)
    dfa.add_edge(2, "b", 0)
    dfa.solid_states = np.array([0, 1, 2])
    retriever = Retriever(dfa, {}, None, max_states=1)
    states = torch.tensor([0, 1, 2])
    neg_dists = torch.tensor([0., -100, 0.])
    states, indices = retriever.transition(states, "a", neg_dists=neg_dists)
    # Follow the transition from 0 -> 1, which has the highest similarity.
    self.assertListEqual(states.tolist(), [1])
    self.assertListEqual(indices.tolist(), [0])

  def test_gen_pointers_from_state(self):
    retriever = Retriever(dfa, inverse_failures, None)
    pointers = list(retriever.gen_pointers_from_state(0))
    self.assertListEqual(pointers, [0, 1, 2])

  def test_gen_pointers_from_state_cutoff(self):
    retriever = Retriever(dfa, inverse_failures, None, max_pointers=2)
    pointers = list(retriever.gen_pointers_from_state(0))
    self.assertListEqual(pointers, [0, 1])

  def test_get_pointers(self):
    retriever = Retriever(dfa, inverse_failures, None)
    states = torch.tensor([0])
    pointers, paired_states = retriever.get_pointers(states)
    self.assertListEqual(pointers.tolist(), [0, 1, 2])
    self.assertListEqual(paired_states.tolist(), [0, 0, 0])

  def test_get_pointers_cutoff(self):
    retriever = Retriever(dfa, inverse_failures, None, max_pointers=2)
    states = torch.tensor([0])
    pointers, paired_states = retriever.get_pointers(states)
    self.assertListEqual(pointers.tolist(), [0, 1])
    self.assertListEqual(paired_states.tolist(), [0, 0])

  def test_get_pointers_solid_only(self):
    retriever = Retriever(dfa, {}, None, solid_only=True)
    states = torch.LongTensor([3])
    pointers, paired_states = retriever.get_pointers(states)
    self.assertListEqual(pointers.tolist(), [])
    self.assertListEqual(paired_states.tolist(), [])

  def test_solidify(self):
    retriever = Retriever(dfa, {}, None)
    states = torch.LongTensor([-1, 1, 2])
    knns = torch.LongTensor([0, 1, 2])
    solid_states = retriever.solidify(states, knns)
    self.assertListEqual(solid_states.tolist(), [0, 1, 2])
