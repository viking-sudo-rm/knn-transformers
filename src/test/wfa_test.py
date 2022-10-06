from unittest import TestCase

from src import wfa
from src import semiring


class WFATest(TestCase):

  sr = semiring.BooleanSemiring()

  def test_recognize_abstar(self):
    aut = wfa.WFA(semiring.PlusTimesSemiring())
    q0 = aut.new_state(1.)
    q1 = aut.new_state(0.)
    aut.add_edge(q0, "a", q1, 1.)
    aut.add_edge(q1, "b", q0, 1.)
    abab = aut.forward("abab")
    self.assertEqual(abab, 1.)
    abb = aut.forward("abb")
    self.assertEqual(abb, 0.)

  def test_string_transduction(self):
    aut = wfa.WFA(semiring.StringSemiring())
    q0 = aut.new_state(["q0"])
    q1 = aut.new_state(["q1"])
    aut.add_edge(q0, "a", q1, ["aa"])
    aut.add_edge(q1, "b", q0, ["c"])
    abab = aut.forward("abab")
    self.assertEqual(abab, ["aacaacq0"])
    abb = aut.forward("abb")
    self.assertEqual(abb, [])

  def test_remove_edge(self):
    dfa = wfa.WFA(self.sr)
    dfa.add_edge(dfa.new_state(), "a", dfa.new_state())
    dfa.add_edge(0, "b", 1)
    dfa.add_edge(1, "b", 1)
    dfa.remove_edge(0, "a")
    self.assertDictEqual(dfa.transitions,
      {
        (0, "b"): (1, None),
        (1, "b"): (1, None),
      }
    )

  def test_failures(self):
    dfa = wfa.WFA(self.sr)
    dfa.add_edge(dfa.new_state(True), "a", dfa.new_state())
    dfa.add_edge(0, "b", dfa.new_state(True))
    dfa.failures[1] = 0
    self.assertEqual(dfa.forward(""), True)
    self.assertEqual(dfa.forward("a"), False)
    self.assertEqual(dfa.forward("aa"), False)
    self.assertEqual(dfa.forward("ab"), True)

  def test_failures_cycle(self):
    dfa = wfa.WFA(self.sr)
    dfa.new_state(True)
    dfa.failures[0] = 0
    # On a cycle, we should avoid an infinite loop and consume the token.
    self.assertEqual(dfa.forward("b"), True)
