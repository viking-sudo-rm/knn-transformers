from unittest import TestCase

from src import wfa
from src import semiring


class WFATest(TestCase):

  n_states = 10

  def test_recognize_abstar(self):
    aut = wfa.WFA(self.n_states)
    q0 = aut.add_state(1.)
    q1 = aut.add_state(0.)
    aut.add_edge(q0, "a", q1)
    aut.add_edge(q1, "b", q0)
    abab = aut.transition("abab")
    self.assertEqual(abab, q0)
    abb = aut.transition("abb")
    self.assertEqual(abb, -1)
  
  def test_add_state(self):
    dfa = wfa.WFA(2)
    dfa.add_state(1)
    self.assertListEqual(list(dfa.weights), [1, -1])

  def test_remove_edge(self):
    dfa = wfa.WFA(self.n_states)
    dfa.add_edge(dfa.add_state(), "a", dfa.add_state())
    dfa.add_edge(0, "b", 1)
    dfa.add_edge(1, "b", 1)
    dfa.remove_edge(0, "a")
    self.assertEqual(dfa.transitions, [[("b", 1)], [("b", 1)]])

  def test_failures(self):
    dfa = wfa.WFA(self.n_states, failures=True)
    dfa.add_edge(dfa.add_state(True), "a", dfa.add_state())
    dfa.add_edge(0, "b", dfa.add_state(True))
    dfa.failures[1] = 0
    self.assertEqual(dfa.transition(""), 0)
    self.assertEqual(dfa.transition("a"), 1)
    self.assertEqual(dfa.transition("aa"), 1)
    self.assertEqual(dfa.transition("ab"), 2)

  def test_failures_cycle(self):
    dfa = wfa.WFA(self.n_states, failures=True)
    dfa.add_state(True)
    dfa.failures[0] = 0
    # On a cycle, we should avoid an infinite loop and consume the token.
    self.assertEqual(dfa.transition("b"), 0)

  def test_forward(self):
    dfa = wfa.WFA(self.n_states)
    dfa.add_state(True)
    dfa.add_state(False)
    dfa.add_edge(0, "a", 1)
    dfa.add_edge(1, "a", 0)
    self.assertEqual(dfa.forward(""), True)
    self.assertEqual(dfa.forward("a"), False)
    self.assertEqual(dfa.forward("aa"), True)
