from typing import List

import numpy as np

from cthree.Optimisable import Optimisable


class Drive(Optimisable):
    """
    Represents a time-dependent drive on a subsystem. This can for example be a microwave or flux drive.
    """

    def getMatrix(self, annihilationOperator: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Return the matrix representation of the drive. The dimension is given by the Hamiltonian to which this drive
        is attached.

        Args:
            annihilationOperator: operator of the subsystem to which this drive is attached
            t (np.ndarray): Vector of time samples

        Returns:
            np.ndarray: matrix of shape [t, n, n]  with t: time, n: hilbert space dimension
        """
        raise NotImplementedError()

    def gradient(self, annihilationOperator: np.ndarray, t: np.ndarray) -> List[np.ndarray]:
        """
        Return the gradient of the matrix representation of the Hamiltonian with respect to each parameter as a list.

        Args:
            annihilationOperator: operator of the subsystem to which this drive is attached
            t (np.ndarray): Vector of time samples

        Returns:
            List[np.ndarray]: List of matrices of shape [t, n, n]  with t: time, n: hilbert space dimension
        """
        raise NotImplementedError()

    @staticmethod
    def _repeat(M: np.ndarray, num: int) -> np.ndarray:
        """
        Utility function that repeats the matrix M for each timestep in the times array. Returns an array with shape
        [t, n, m] where t is the number of time steps and M is a n times m matrix.
        """
        return M.reshape((1,) + M.shape).repeat(num, axis=0)

