from typing import List

import numpy as np

from cthree.Quantity import Quantity
from cthree.model.Drive import Drive
from cthree.model.Hamiltonian import Hamiltonian


class Qubit(Hamiltonian):
    """
    Hamiltonian of a single qubit frequency/2 * sigma_z.

    The implementation uses the convention of having the excited state of the qubit as the first entry in the state. If
    you need a two-level system that is compatible with the projection of a higher-dimensional system (ground state as
    first entry), use a resonator and restrict its dimension to 2.
    """
    __frequency: Quantity

    def __init__(self, frequency: Quantity, drives: List[Drive] = None):
        super().__init__(drives)
        self.__frequency = frequency

    def getFrequency(self) -> Quantity:
        return self.__frequency

    def setFrequency(self, frequency: Quantity) -> None:
        self.__frequency = frequency

    def getParameters(self) -> List[Quantity]:
        return [self.__frequency] + self._getDriveParameters()

    def dimension(self) -> int:
        return 2

    def getMatrix(self, t: np.ndarray) -> np.ndarray:
        H = 0.5 * self.__frequency.getValue() * np.diag([1.0, -1.0])
        return self._repeatInTime(H, t)

    def gradient(self, t: np.ndarray) -> List[np.ndarray]:
        # derivative wrt the frequency
        H = 0.5 * np.diag([1.0, -1.0])
        return [self._repeatInTime(H, t)]
