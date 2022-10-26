from unittest import TestCase
import torch

from src.state_manager import StateManager
from src.semiring import PointerSemiring
from src.retriever import Retriever
from src.wfa import WFA


dfa = WFA(PointerSemiring())
dfa.new_state(weight=[0])
dfa.new_state(weight=[0])
dfa.new_state(weight=[1])
dfa.add_edge(0, "a", 1)
dfa.add_edge(1, "a", 0)
dfa.add_edge(1, "a", 2)
dfa.failures = {0: 0, 1: 0, 2: 0}

solid_states = [0, 2]
retriever = Retriever(dfa, {}, None)


class StateManagerTest(TestCase):

    def test_transition_basic(self):
        manager = StateManager(solid_states, retriever)
        manager.transition("a")
        self.assertSetEqual(manager.states, set([1]))

    def test_transition_colliding(self):
        manager = StateManager(solid_states, retriever, {0, 1})
        manager.transition("a")
        self.assertSetEqual(manager.states, {1, 2})
    
    def test_transition_failing(self):
        manager = StateManager(solid_states, retriever, {0, 1, 2})
        manager.transition("b")
        self.assertSetEqual(manager.states, {0})

    def test_get_pointers(self):
        manager = StateManager(solid_states, retriever, {0, 1, 2})
        pointers = manager.get_pointers()
        self.assertListEqual(pointers, [0, 1])

    def test_get_pointers_solid_only(self):
        manager = StateManager(solid_states, retriever, {1, 2}, solid_only=True)
        pointers = manager.get_pointers()
        self.assertListEqual(pointers, [1])

    def test_add_pointers_no_clear(self):
        manager = StateManager(solid_states, retriever, clear=False)
        manager.add_pointers(torch.tensor([1]))
        self.assertSetEqual(manager.states, set([0, 2]))

    def test_add_pointers_with_limit_no_clear(self):
        manager = StateManager(solid_states, retriever, {1}, max_states=2, clear=False)
        manager.add_pointers([0, 1])
        self.assertSetEqual(manager.states, {0, 1})

    def test_add_pointers_no_limit(self):
        manager = StateManager(solid_states, retriever, {1})
        manager.add_pointers([0, 1])
        self.assertSetEqual(manager.states, {0, 2})

    def test_add_pointers(self):
        manager = StateManager(solid_states, retriever, {1}, max_states=1)
        manager.add_pointers([0, 1])
        self.assertSetEqual(manager.states, {0})
