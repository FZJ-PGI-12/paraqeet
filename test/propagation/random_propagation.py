"""Test the random propagation model."""

import numpy as np
from scipy.stats import unitary_group

from cthree.propagation.propagation import Propagation
from cthree.quantity import Quantity


class RandomPropagation(Propagation):
    """Mock random propagation implementation.

    Returns random state vectors, density matrices, or propagators.

    Parameters
    ----------
    dimension : int
        Hilbert space size for the generated states.
    generateMatrices : bool, default=False
        Whether to generate matrices instead of vectors.
    autoUpdate : bool, default=True
        Whether to return a new random state at every call of propagate.
        If false, propagate will return the same state until update was called.
    """

    __dimension: int
    __createMatrices: bool
    __autoUpdate: bool
    __state: np.ndarray

    def __init__(
        self,
        dimension: int,
        generateMatrices: bool = False,
        autoUpdate: bool = True,
    ):
        super().__init__(None)
        self.__dimension = dimension
        self.__createMatrices = generateMatrices
        self.__autoUpdate = autoUpdate
        self.update()

    def get_parameters(self) -> list[Quantity]:
        return []

    def set_initial_state(self, state: np.ndarray):
        """Set the initial state of the system.

        Set it to the given state.

        Parameters
        ----------
        state : numpy.ndarray
            Given state to set as the initial state.

        """
        pass

    def propagate(self, time: np.ndarray) -> np.ndarray:
        """Propagate the system through time.

        Parameters
        ----------
        time : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Returns the updated state of the system.

        """
        if self.__autoUpdate:
            self.update()
        return np.array([self.__state] * len(time))

    def update(self) -> None:
        """Update the state on propagation.

        Makes sure that the next call to propagate will return a
        new random state.

        """
        if self.__createMatrices:
            # generate a random density matrix by rotating a
            # random diagonal matrix
            rho = np.diag(np.random.random(self.__dimension))
            rho /= np.trace(rho)
            U = unitary_group.rvs(self.__dimension)
            self.__state = np.conjugate(U.T) @ rho @ U
        else:
            # generate a random state vector
            state = np.random.random((self.__dimension, 1)) + 1j * np.random.random((self.__dimension, 1))
            self.__state = state / np.sqrt(np.vdot(state, state))
