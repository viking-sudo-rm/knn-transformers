"""Metrics classes."""

from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

from src.state_manager import StateManager


class Metrics(metaclass=ABCMeta):

    @abstractmethod
    def update(self,
               sm: StateManager,
               pointers: list[int],
               cur_knns: list[int]) -> None:
        return NotImplemented

    @abstractmethod
    def get_metrics_dict(self) -> dict:
        return NotImplemented

    def action(self, path: str) -> None:
        return


class CountMetrics(Metrics):

    def __init__(self,
                 n_total: int = 0,
                 n_knns: int = 0,
                 n_pointers: int = 0,
                 n_empty: int = 0,
                 n_states: int = 0,
                ):
        self.n_total = n_total
        self.n_knns = n_knns
        self.n_pointers = n_pointers
        self.n_empty = n_empty
        self.n_states = n_states

    def update(self, sm, pointers, cur_knns) -> None:
        self.n_total += 1
        self.n_knns += len(cur_knns)
        self.n_pointers += len(pointers)
        self.n_states += len(sm.states)
        if len(sm.states) == 0:
            self.n_empty += 1
    
    def get_metrics_dict(self) -> dict:
        return {
            "n_knns": self.n_knns / self.n_total,
            "n_states": self.n_pointers / self.n_total,
            "n_pointers": self.n_pointers / self.n_total,
            "n_empty": self.n_empty / self.n_total,
        }


class PlotMetrics(Metrics):

    def __init__(self):
        self.n_knns = []
        self.n_pointers = []
        self.n_states = []

    def update(self, sm, pointers, cur_knns):
        self.n_knns.append(len(cur_knns))
        self.n_pointers.append(len(pointers))
        self.n_states.append(len(sm.states))

    def get_metrics_dict(self) -> dict:
        return {
            "n_knns": np.mean(self.n_knns),
            "n_states": np.mean(self.n_states),
            "n_pointers": np.mean(self.n_pointers),
        }

    def action(self, path: str) -> None:
        plt.plot(self.n_pointers, label="pointers")
        plt.plot(self.n_states, label="states")
        plt.plot(self.n_knns, label="knns")
        plt.xlabel("Steps")
        plt.ylabel("Count")
        plt.savefig(path)
