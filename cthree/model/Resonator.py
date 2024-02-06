from typing import List, Tuple

import numpy as np

from cthree.Quantity import Quantity
from cthree.model.Drive import Drive
from cthree.model.Hamiltonian import Hamiltonian


class Resonator(Hamiltonian):
    """
    Hamiltonian of a harmonic oscillator. The only optimisable parameter is the frequency.
    """
    __dimension: int
    __frequency: Quantity
    __annihilationOp: np.ndarray
    __numOp: np.ndarray

    def __init__(self, dimension: int, frequency: Quantity, drives: List[Drive] = None):
        super().__init__(drives=drives)
        self.__dimension = dimension
        self.__frequency = frequency
        self.__annihilationOp = np.sqrt(np.diag(np.arange(1, dimension, dtype=np.float64), k=1))
        self.__numOp = self.__annihilationOp.T @ self.__annihilationOp

    def dimension(self):
        return self.__dimension

    def getFrequency(self) -> Quantity:
        return self.__frequency

    def setFrequency(self, frequency: Quantity) -> None:
        self.__frequency = frequency

    def getParameters(self) -> List[Quantity]:
        return [self.__frequency] + self._getDriveParameters()

    def getMatrix(self, t: np.ndarray) -> np.ndarray:
        H = self.__frequency * self.__numOp
        return self._repeatInTime(H, t) + self._getDriveMatrix(self.__annihilationOp, t)

    def gradient(self, t: np.ndarray) -> List[np.ndarray]:
        # Derivative wrt to the frequency
        grad = self.__numOp.reshape(self.__numOp.shape + (1,))
        return [self._repeatInTime(grad, t)] + self._getDriveGradients(self.__annihilationOp, t)
