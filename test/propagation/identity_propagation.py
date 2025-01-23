"""Test the identity propagation model."""

import numpy as np

from cthree.propagation.propagation import Propagation


class IdentityPropagation(Propagation):
    """Mock identity propagation implementation.

    Returns the initial state as the target state.
    """

    __state: np.ndarray

    def __init__(self):
        super().__init__(None)

    def set_initial_state(self, state: np.ndarray):
        """Set the initial state of the system.

        Set it to the given state argument.

        Parameters
        ----------
        state : numpy.ndarray
            Given state to be set as the initial state.

        """
        self.__state = state

    def propagate(self, time: np.ndarray) -> np.ndarray:
        """Get the propagated state across the timestamps.

        Parameters
        ----------
        time : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Returns the propagated values of the state across timestamps.

        """
        return np.array([self.__state] * len(time))
