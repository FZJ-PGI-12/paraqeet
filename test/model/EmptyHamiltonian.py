"""Class definition of the empty Hamiltonian for testing."""

import numpy as np

from cthree.model.Hamiltonian import Hamiltonian


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

    def get_matrix(self, t: np.ndarray) -> np.ndarray:
        """Get the matrix representation of the Hamiltonian.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            The matrix representation of the Hamiltonian.

        """
        return np.zeros((len(t), self.__dimension, self.__dimension))
