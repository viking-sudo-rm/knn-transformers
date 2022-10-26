from .retriever import Retriever
from .wfa import WFA


class StateManager:

    """
    Logic for maintaining a set of active states, retrieving pointers from it, and adding states based on pointers.

    Args:
        solid_states: A mapping from pointers to states.
        retriever: An object for retrieving pointers from states.

    TODO:
        Replace states with a list instead of set? We might want to upweight states occuring more than once.
    """

    def __init__(self,
                 solid_states: list[int],
                 retriever: Retriever,
                 states: set[int] = None,
                 solid_only: bool = False,
                 max_states: int = -1,
                 clear: bool = True,
                 add_initial: bool = True,
                ):
        self.dfa: WFA = retriever.dfa
        self.solid_states = solid_states
        self.retriever = retriever
        self.solid_only = solid_only
        self.max_states = max_states
        self.clear = clear

        if states:
            self.states = states
        elif add_initial:
            self.states = set([self.dfa.initial])
        else:
            self.states = set()

    def transition(self, token) -> None:
        queue = list(self.states)
        for state in queue:
            self.states.remove(state)
        for state in queue:
            next_state, _ = self.dfa.next_state(state, token)
            # Handle the case where the DFA does not have failure transitions.
            if next_state is not None:
                self.states.add(next_state)

    def get_pointers(self):
        if not self.solid_only:
            return [ptr for ptr, _ in self.retriever.gen_pointers(self.states)]  # In the range [0, n].
        else:
            return [ptr for ptr, state in self.retriever.gen_pointers(self.states) if self.solid_states[ptr] == state]

    def add_pointers(self, pointers) -> None:
        """Add pointers from a list to the state manager.
        
        Assumes pointers are sorted by priority."""
        if self.clear:
            self.states.clear()

        if self.max_states == -1:
            self.states.update(self.solid_states[ptr] for ptr in pointers)
            return

        for ptr in pointers:
            if len(self.states) >= self.max_states:
                break
            state = self.solid_states[ptr]
            self.states.add(state)
