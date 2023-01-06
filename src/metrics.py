"""Metrics classes."""

from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class Metrics(metaclass=ABCMeta):

    @abstractmethod
    def update(self,
               states: list[int],
               pointers: list[int]) -> None:
        return NotImplemented

    @abstractmethod
    def get_metrics_dict(self) -> dict:
        return NotImplemented

    def action(self, path: str) -> None:
        return


class CountMetrics(Metrics):

    def __init__(self,
                 n_total: int = 0,
                 n_states: int = 0,
                 n_pointers: int = 0,
                 n_empty: int = 0,
                ):
        self.n_total = n_total
        self.n_states = n_states
        self.n_pointers = n_pointers
        self.n_empty = 0

    def update(self, states, pointers) -> None:
        self.n_total += 1
        self.n_states += len(states)
        self.n_pointers += len(pointers)
        if len(states) == 0:
            self.n_empty += 1
    
    def get_metrics_dict(self) -> dict:
        return {
            "n_states": self.n_states / self.n_total,
            "n_pointers": self.n_pointers / self.n_total,
            "n_empty": self.n_empty / self.n_total,
        }


class PlotMetrics(Metrics):

    def __init__(self):
        self.n_pointers = []
        self.n_states = []

    def update(self, states, pointers):
        self.n_pointers.append(len(pointers))
        self.n_states.append(len(sm.states))

    def get_metrics_dict(self) -> dict:
        return {
            "n_states": np.mean(self.n_states),
            "n_pointers": np.mean(self.n_pointers),
        }

    def action(self, path: str) -> None:
        plt.plot(self.n_pointers, label="pointers")
        plt.plot(self.n_states, label="states")
        plt.xlabel("Steps")
        plt.ylabel("Count")
        plt.savefig(path)
