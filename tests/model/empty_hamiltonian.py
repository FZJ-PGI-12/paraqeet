"""Class definition of the empty Hamiltonian for testing."""

import numpy as np
from paraqeet.quantity import Array
from paraqeet.model.hamiltonian import Hamiltonian
from paraqeet.quantity import Quantity


class EmptyHamiltonian(Hamiltonian):
    """A Hamiltonian that is filled with zeros for all time steps.

    Parameters
    ----------
    dimension : int
        The dimension for the representation of the Hamiltonian.
    """

    __dimension: int

    def __init__(self, dimension: int):
        super().__init__([])

        self.__dimension = dimension

    def get_matrix(self, t: Array) -> Array:
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
        return np.zeros((len(t), self.__dimension, self.__dimension))

    def get_parameters(self) -> list[Quantity]:
        """ """
        return []

    # TODO: implement abstract methods from Hamiltonian
    def dimension(self) -> int:
        raise NotImplementedError("Method not implemented yet.")

    # TODO: implement abstract methods from Hamiltonian
    def get_matrix_one_time(self, t: Array) -> Array:
        raise NotImplementedError("Method not implemented yet.")

    # TODO: implement abstract methods from Hamiltonian
    def gradient_one_time(self, t: Array) -> Array:
        raise NotImplementedError("Method not implemented yet.")

    # TODO: implement abstract methods from Hamiltonian
    def get_collapseops(self) -> list[tuple[Array, Array]]:
        raise NotImplementedError("Method not implemented yet.")
