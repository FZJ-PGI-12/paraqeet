from typing import List

import numpy as np

from cthree.Optimisable import Optimisable
from cthree.Quantity import Quantity
from cthree.model.Drive import Drive


class Hamiltonian(Optimisable):
    """
    Matrix representation of a Hamiltonian. Implementations can contain subsystems, couplings, and drive lines and have
    to take care of frame transformations. Derived classes need to implement the functions getMatrix, gradient, and
    dimension.
    """
    _drives: List[Drive]

    def __init__(self, drives=None):
        self._drives = [d for d in drives if d is not None] or []

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

    def getDrives(self) -> List[Drive]:
        return self._drives

    def _getDriveParameters(self) -> List[Quantity]:
        """
        Returns the combined list of parameters from all drives.
        """
        params = []
        for d in self._drives:
            params += d.getParameters()
        return params

    def _getDriveMatrix(self, annihilationOperator: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Returns the sum of all drives in matrix form. This function can be used be Hamiltonian implementations for
        including the drive.
        """
        dim = self.dimension()
        M = np.zeros((dim, dim))
        for drive in self._drives:
            M += drive.getMatrix(annihilationOperator, t)
        return M

    def _getDriveGradients(self, annihilationOperator: np.ndarray, t: np.ndarray) -> List[np.ndarray]:
        """
        Returns the gradients of all drives. This function can be used be Hamiltonian implementations for including
        the drive gradients.
        """
        dim = self.dimension()
        driveGrads = []
        for drive in self._drives:
            driveGrads += drive.gradient(annihilationOperator, t)
        return driveGrads

    @staticmethod
    def _repeat(M: np.ndarray, num: int) -> np.ndarray:
        """
        Utility function that repeats the matrix M for each timestep in the times array. Returns an array with shape
        [t, n, m] where t is the number of time steps and M is a n times m matrix.
        """
        return M.reshape((1,) + M.shape).repeat(num, axis=0)
