"""Class definition of the empty Hamiltonian for testing."""

import numpy as np

from paraqeet.hamiltonian.hamiltonian import Hamiltonian
from paraqeet.quantity import Array, Quantity


class EmptyHamiltonian(Hamiltonian):
    """A Hamiltonian that is filled with zeros for all time steps."""

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The dimension for the representation of the Hamiltonian.
        """
        super().__init__([])

        self._dimension = dimension

    def get_value(self, times: Array) -> Array:
        return np.zeros((len(times), self._dimension, self._dimension))

    def get_gradient(self, times: Array) -> Array:
        return np.zeros((len(times), 0, self._dimension, self._dimension))

    def get_parameters(self) -> list[Quantity]:
        return []

    def dimension(self) -> int:
        return self._dimension
