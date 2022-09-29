import unittest

from src.wfa import WFA
from src.semiring import BooleanSemiring
from src.retriever_builder import RetrieverBuilder


class RetrieverBuilderTest(unittest.TestCase):

    def test_build_factor_lengths_solid(self):
        dfa = WFA(BooleanSemiring())
        states = [dfa.new_state() for _ in range(10)]
        for idx in range(1, len(states)):
            dfa.add_edge(states[idx - 1], "a", states[idx])
        for idx in range(5, len(states), 3):
            dfa.add_edge(states[idx - 3], "b", states[idx])
        factor_lengths = RetrieverBuilder.build_factor_lengths(dfa)
        self.assertDictEqual(factor_lengths,
                             {0: 0, 1: 1, 2: 2, 3: 3, 5: 3, 4: 4, 6: 4, 8: 4, 7: 5, 9: 5})
