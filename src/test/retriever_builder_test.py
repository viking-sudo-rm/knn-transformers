import unittest

from src.wfa import WFA
from src.semiring import BooleanSemiring
from src.retriever_builder import RetrieverBuilder
from src.suffix_dfa_builder import SuffixDfaBuilder


solid_dfa = WFA(BooleanSemiring())
states = [solid_dfa.new_state() for _ in range(10)]
for idx in range(1, len(states)):
    solid_dfa.add_edge(states[idx - 1], "a", states[idx])
for idx in range(5, len(states), 3):
    solid_dfa.add_edge(states[idx - 3], "b", states[idx])

builder = SuffixDfaBuilder()
builder.build("ababa")
builder.add_failures()
suffix_dfa = builder.dfa


class RetrieverBuilderTest(unittest.TestCase):

    def test_build_factor_lengths_solid(self):
        factor_lengths = RetrieverBuilder.build_factor_lengths(solid_dfa)
        self.assertDictEqual(factor_lengths,
                             {0: 0, 1: 1, 2: 2, 3: 3, 5: 3, 4: 4, 6: 4, 8: 4, 7: 5, 9: 5})

    def test_build_inverse_failures_solid(self):
        inverse_failures = RetrieverBuilder.build_inverse_failures(solid_dfa)
        self.assertDictEqual(inverse_failures, {})

    def test_build_factor_lengths_suffix(self):
        factor_lengths = RetrieverBuilder.build_factor_lengths(suffix_dfa)
        self.assertDictEqual(factor_lengths, {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4})

    def test_build_inverse_failures_suffix(self):
        inverse_failures = RetrieverBuilder.build_inverse_failures(suffix_dfa)
        self.assertEqual(inverse_failures, {0: [1, 2], 1: [3], 2: [4], 3: [5]})