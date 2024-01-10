from typing import List

import numpy as np
from cthree.Quantity import Quantity

from cthree.signal.Generator import Generator
from cthree.Optimisable import Optimisable


class Hamiltonian(Optimisable):
    """
    Matrix representation of a Hamiltonian.
    Contains subsystems, couplings, and drive lines.
    Takes care of frame transformations.
    """

    __subsystems: List
    __couplings: List
    __drives: List
    __generator: Generator | None

    def __init__(
        self,
        subsystems: List,
        couplings: List = [],
        drives: List = [],
        generator: Generator | None = None,
    ):
        self.__subsystems = subsystems
        self.__couplings = couplings
        self.__drives = drives
        self.__generator = generator

    def getDrives(self) -> List[np.ndarray]:
        return self.__drives

    def getMatrix(self, t: np.ndarray) -> np.ndarray:
        """
        Return the matrix representation of the Hamiltonian.

        Args:
            t (np.ndarray): Vector of time samples

        Returns:
            np.ndarray: Hamiltonian of shape [t, n, n]  with t: time, n: hilbert space
        """
        if self.__generator:
            sig = self.__generator.generateSignal(t)
        return self.__subsystems[0] + sig * self.__drives[0]

    def getParameters(self) -> List[Quantity]:
        return []

    def gradient(self, t: np.ndarray) -> List[np.ndarray]:
        """
        Return the gradient of each parameter as a list.
        """
        grads = self.__generator.generateSignalGradient(t)
        return [g * self.getDrives()[0] for g in grads]
