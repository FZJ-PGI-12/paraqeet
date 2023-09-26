import numpy as np
from scipy.stats import unitary_group

from cthree.propagation.Propagation import Propagation


class RandomPropagation(Propagation):
    """
    Mock propagation implementation that returns random state vectors or density matrices.
    """
    __dimension: int
    __mixedState: bool
    __autoUpdate: bool
    __state: np.ndarray

    def __init__(self, dimension: int, mixedState: bool = False, autoUpdate: bool = True):
        """
        :param dimension: Hilbert space size for the generated states
        :param mixedState: whether to generate density matrices instead of vectors
        :param autoUpdate: Whether to return a new random state at every call of propagate. If false, propagate will
                           return the same state until update was called.
        """
        super().__init__(None)
        self.__dimension = dimension
        self.__mixedState = mixedState
        self.__autoUpdate = autoUpdate
        self.update()

    def propagate(self) -> np.ndarray:
        if self.__autoUpdate:
            self.update()
        return self.__state

    def update(self) -> None:
        """
        Makes sure that the next call to propagate will return a new random state.
        """
        if self.__mixedState:
            # generate a random density matrix by rotating a random diagonal matrix
            rho = np.diag(np.random.random(self.__dimension))
            rho /= np.trace(rho)
            U = unitary_group.rvs(self.__dimension)
            self.__state = np.conjugate(U.T) @ rho @ U
        else:
            # generate a random state vector
            state = np.random.random(self.__dimension) + 1j * np.random.random(self.__dimension)
            self.__state = state / np.sqrt(np.vdot(state, state))
