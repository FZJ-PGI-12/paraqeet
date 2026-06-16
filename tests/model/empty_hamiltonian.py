"""Class definition of the empty Hamiltonian for testing."""

from typing import override

import numpy as np

from paraqeet.model.system import OpenSystem
from paraqeet.quantity import Array, Quantity


class EmptySystem(OpenSystem):
    """A Hamiltonian that is filled with zeros for all time steps.

    Parameters
    ----------
    dimension : int
        The dimension for the representation of the Hamiltonian.
    """

    _dimension: int

    def __init__(self, dimension: int):
        super().__init__([])

        self._dimension = dimension

    @override
    def get_value(self, times: Array) -> Array:
        return np.zeros((len(times), self._dimension, self._dimension))

    @override
    def get_gradient(self, times: Array) -> Array:
        return np.zeros((len(times), 0, self._dimension, self._dimension))

    @override
    def get_parameters(self) -> list[Quantity]:
        """ """
        return []

    # TODO: implement abstract methods from Hamiltonian
    def dimension(self) -> int:
        raise NotImplementedError("Method not implemented yet.")

    # TODO: implement abstract methods from Hamiltonian
    def get_jump_operators(self) -> list[tuple[Array, Array]]:
        raise NotImplementedError("Method not implemented yet.")
