from collections import defaultdict, deque

from .retriever import Retriever


class RetrieverBuilder:

    @classmethod
    def build(cls, dfa, **kwargs):
        min_factor_length = kwargs.get("min_factor_length", 0)
        inverse_failures = cls.build_inverse_failures(dfa)
        factor_lengths = cls.build_factor_lengths(dfa) if min_factor_length > 0 else None
        return Retriever(dfa, inverse_failures, factor_lengths, **kwargs)

    @staticmethod
    def build_inverse_failures(dfa):
        inverse_failures = defaultdict(list)
        for state, fail_state in dfa.failures.items():
            if state is None or fail_state is None:
                continue
            inverse_failures[fail_state].append(state)
        inverse_failures[dfa.initial].remove(dfa.initial)
        return inverse_failures

    @staticmethod
    def build_factor_lengths(dfa):
        """Find shortest path to each state by breadth-first search."""
        factor_lengths = {dfa.initial: 0}
        queue = deque([dfa.initial])
        while queue:
            state = queue.popleft()
            length = factor_lengths[state]
            for token in dfa.edges_out[state]:
                next_state, _ = dfa.transitions[state, token]
                # We only care about the minimum-depth occurence of each state.
                if next_state not in factor_lengths:
                    factor_lengths[next_state] = length + 1
                    queue.append(next_state)
        return factor_lengths
