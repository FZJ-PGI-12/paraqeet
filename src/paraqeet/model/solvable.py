from abc import ABC, abstractmethod

from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Array


class System(Optimizable):
    def __init__(self):
        pass

    @abstractmethod
    def get_hamiltonian(self, times: Array) -> Array:
        pass

    @abstractmethod
    def get_hamiltonian_and_gradient(self, times: Array) -> Array:
        pass

    @abstractmethod
    def get_gradient_at_timestep(self, timestep: float) -> Array:
        pass
