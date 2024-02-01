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
    __numOp: np.ndarray

    def __init__(self, dimension: int, frequency: Quantity, drives: List[Drive] = None):
        super().__init__(drives=drives)
        self.__dimension = dimension
        self.__frequency = frequency
        self.__numOp = np.diag(np.arange(0, self.__dimension))

    def dimension(self):
        return self.__dimension

    def getFrequency(self) -> Quantity:
        return self.__frequency

    def setFrequency(self, frequency: Quantity) -> None:
        self.__frequency = frequency

    def getParameters(self) -> List[Quantity]:
        return [self.__frequency]

    def getMatrix(self, t: np.ndarray) -> np.ndarray:
        H = self.__frequency * self.__numOp
        return self._repeatInTime(H, t) + self._getDriveMatrix(t)

    def gradient(self, t: np.ndarray) -> List[np.ndarray]:
        # Derivative wrt to the frequency
        grad = self.__numOp.reshape(self.__numOp.shape + (1,))
        return [self._repeatInTime(grad, t)] + self._getDriveGradients(t)
