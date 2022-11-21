from .retriever import Retriever
from .wfa import WFA


class StateManager:

    """
    Logic for maintaining a set of active states, retrieving pointers from it, and adding states based on pointers.

    Args:
        solid_states: A mapping from pointers to states.
        retriever: An object for retrieving pointers from states.

    TODO:
        Refactor to be stateless, add initialize method?
    """

    def __init__(self,
                 retriever: Retriever,
                 states: list[int] = None,
                 solid_only: bool = False,
                 max_states: int = -1,
                 add_initial: bool = True,
                ):
        self.retriever = retriever
        self.dfa: WFA = retriever.dfa
        self.solid_states = self.dfa.solid_states
        self.solid_only = solid_only
        self.max_states = max_states

        if states:
            self.states = states
        elif add_initial:
            self.states = [self.dfa.initial]
        else:
            self.states = []

    def get_pointers(self):
        if not self.solid_only:
            return [ptr for ptr, _ in self.retriever.gen_pointers(self.states)]  # In the range [0, n].
        else:
            return [ptr for ptr, state in self.retriever.gen_pointers(self.states) if self.solid_states[ptr] == state]

    def add_pointers(self, pointers) -> None:
        """Add pointers from a list to the state manager.
        
        Assumes pointers are sorted by priority."""
        if self.max_states != -1:
            pointers = pointers[:self.max_states]
        self.states = [self.solid_states[ptr] for ptr in pointers]

    def transition(self, token) -> None:
        self.states = [self.dfa.next_state(q, token) for q in self.states]
        self.states = [q for q in self.states if q is not None]
