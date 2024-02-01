from typing import List

import numpy as np

from cthree.Optimisable import Optimisable


class Hamiltonian(Optimisable):
    """
    Matrix representation of a Hamiltonian.
    Implementations can contain subsystems, couplings, and drive lines and have to take care of frame transformations.
    """

    def dimension(self) -> int:
        """
        Returns the dimension of the Hilbert space of this Hamiltonian.
        """
        raise NotImplementedError()

    def getMatrix(self, t: np.ndarray) -> np.ndarray:
        """
        Return the matrix representation of the Hamiltonian.

        Args:
            t (np.ndarray): Vector of time samples

        Returns:
            np.ndarray: Hamiltonian of shape [t, n, n]  with t: time, n: hilbert space
        """
        raise NotImplementedError()

    def gradient(self, t: np.ndarray) -> np.ndarray:
        """
        Return the gradient of the matrix representation of the Hamiltonian with respect to each parameter as a list.
        """
        raise NotImplementedError()

    @staticmethod
    def _repeatInTime(M: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Utility function that repeats the matrix M for each timestep in the times array. Returns an array with shape
        [t, n, m] where t is the number of time steps and M is a n times m matrix.
        """
        return M.reshape((1,) + M.shape).repeat(len(times), axis=0)
