import numpy as np
from scipy.stats import unitary_group

from cthree.propagation.Propagation import Propagation


class RandomPropagation(Propagation):
    """
    Mock propagation implementation that returns random state vectors, density matrices, or propagators.
    """

    __dimension: int
    __createMatrices: bool
    __autoUpdate: bool
    __state: np.ndarray

    def __init__(
        self, dimension: int, generateMatrices: bool = False, autoUpdate: bool = True
    ):
        """
        :param dimension: Hilbert space size for the generated states
        :param generateMatrices: whether to generate matrices instead of vectors
        :param autoUpdate: Whether to return a new random state at every call of propagate. If false, propagate will
                           return the same state until update was called.
        """
        super().__init__(None)
        self.__dimension = dimension
        self.__createMatrices = generateMatrices
        self.__autoUpdate = autoUpdate
        self.update()

    def setInitialState(self, state: np.ndarray):
        pass

    def propagate(self, time: np.ndarray) -> np.ndarray:
        if self.__autoUpdate:
            self.update()
        return np.array([self.__state] * len(time))

    def update(self) -> None:
        """
        Makes sure that the next call to propagate will return a new random state.
        """
        if self.__createMatrices:
            # generate a random density matrix by rotating a random diagonal matrix
            rho = np.diag(np.random.random(self.__dimension))
            rho /= np.trace(rho)
            U = unitary_group.rvs(self.__dimension)
            self.__state = np.conjugate(U.T) @ rho @ U
        else:
            # generate a random state vector
            state = np.random.random((self.__dimension, 1)) + 1j * np.random.random(
                (self.__dimension, 1)
            )
            self.__state = state / np.sqrt(np.vdot(state, state))
