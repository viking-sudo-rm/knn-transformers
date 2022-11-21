from collections import defaultdict, deque

from .retriever import Retriever


class RetrieverBuilder:

    def __init__(self, min_factor_length: int = 0, **kwargs):
        self.min_factor_length = min_factor_length
        self.kwargs = kwargs

    def build(self, dfa):
        # min_factor_length = kwargs.get("min_factor_length", 0)
        inverse_failures = self.build_inverse_failures(dfa)
        factor_lengths = self.build_factor_lengths(dfa) if self.min_factor_length > 0 else None
        return Retriever(dfa,
                         inverse_failures,
                         factor_lengths,
                         min_factor_length=self.min_factor_length,
                         **self.kwargs,
                        )

    @staticmethod
    def build_inverse_failures(dfa):
        if dfa.failures is None:
            return {}
        inverse_failures = defaultdict(list)
        for state, fail_state in enumerate(dfa.failures):
            if state is None or fail_state is None:
                continue
            inverse_failures[fail_state].append(state)
        if dfa.initial in inverse_failures:
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
            # for token in dfa.edges_out[state]:
            for token, next_state in dfa.transitions[state]:
                # next_state, _ = dfa.transitions[state, token]
                # We only care about the minimum-depth occurence of each state.
                if next_state not in factor_lengths:
                    factor_lengths[next_state] = length + 1
                    queue.append(next_state)
        return factor_lengths
