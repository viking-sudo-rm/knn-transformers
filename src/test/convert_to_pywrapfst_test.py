from unittest import TestCase

from src.wfa import WFA
from src.convert_to_pywrapfst import convert_to_pywrapfst


def _arc_to_tuple(arc):
    return arc.ilabel, arc.olabel, arc.nextstate, arc.weight.to_string()

def _weight_to_int(weight):
    return int(weight.to_string())


class ConvertToPyWrapFstTest(TestCase):

    def test_conversion_basic(self):
        dfa = WFA(10)
        dfa.add_state(10)
        dfa.add_state(11)
        dfa.add_edge(0, 96, 1)
        dfa.add_edge(1, 97, 0)
        fst = convert_to_pywrapfst(dfa)
        states = list(fst.states())
        weights = [_weight_to_int(fst.final(q)) for q in states]
        arcs0 = [_arc_to_tuple(arc) for arc in fst.arcs(0)]
        arcs1 = [_arc_to_tuple(arc) for arc in fst.arcs(1)]
        self.assertTrue(fst.verify())
        self.assertListEqual(states, [0, 1])
        self.assertListEqual(weights, [10, 11])
        self.assertListEqual(arcs0, [(96, 96, 1, "0")])
        self.assertListEqual(arcs1, [(97, 97, 0, "0")])

    def test_conversion_failures(self):
        dfa = WFA(10, failures=True)
        dfa.add_state(10)
        dfa.add_state(11)
        dfa.add_edge(0, 96, 1)
        dfa.failures[1] = 0
        fst = convert_to_pywrapfst(dfa)
        states = list(fst.states())
        weights = [_weight_to_int(fst.final(q)) for q in states]
        arcs0 = [_arc_to_tuple(arc) for arc in fst.arcs(0)]
        arcs1 = [_arc_to_tuple(arc) for arc in fst.arcs(1)]
        # FIXME: Set the failure transition token.
        # self.assertTrue(fst.verify())
        self.assertListEqual(states, [0, 1])
        self.assertListEqual(weights, [10, 11])
        self.assertListEqual(arcs0, [(96, 96, 1, "0")])
        self.assertListEqual(arcs1, [(-1, -1, 0, "0")])
