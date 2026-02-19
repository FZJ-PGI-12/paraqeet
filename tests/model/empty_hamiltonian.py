"""Class definition of the empty Hamiltonian for testing."""

import numpy as np

from paraqeet.model.system import OpenSystem
from paraqeet.quantity import Array, Quantity


class EmptyHamiltonian(OpenSystem):
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

    def get_hamiltonian(self, times: Array) -> Array:
        """Get the matrix representation of the Hamiltonian.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            The matrix representation of the Hamiltonian.

        """
        return np.zeros((len(times), self._dimension, self._dimension))

    def get_parameters(self) -> list[Quantity]:
        """ """
        return []

    # TODO: implement abstract methods from Hamiltonian
    def dimension(self) -> int:
        raise NotImplementedError("Method not implemented yet.")

    # TODO: implement abstract methods from Hamiltonian
    def get_hamiltonian_at_timestep(self, timestep: Array) -> Array:
        raise NotImplementedError("Method not implemented yet.")

    # TODO: implement abstract methods from Hamiltonian
    def get_gradient_at_timestep(self, t: Array) -> Array:
        raise NotImplementedError("Method not implemented yet.")

    # TODO: implement abstract methods from Hamiltonian
    def get_collapseops(self) -> list[tuple[Array, Array]]:
        raise NotImplementedError("Method not implemented yet.")
