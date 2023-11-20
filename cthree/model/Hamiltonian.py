from typing import List, Optional

import numpy as np

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
    __generator: Generator

    def __init__(
        self,
        subsystems: List,
        couplings: List = [],
        drives: List = [],
        generator: Optional[Generator] = None,
    ):
        self.__subsystems = subsystems
        self.__couplings = couplings
        self.__drives = drives
        self.__generator = generator

    def getMatrix(self, t: np.ndarray) -> np.ndarray:
        """
        Return the matrix representation of the Hamiltonian.

        Args:
            t (np.ndarray): Vector of time samples

        Returns:
            np.ndarray: Hamiltonian of shape [t, n, n]  with t: time, n: hilbert space
        """
        sig = self.__generator.generateSignal(t)
        return self.__subsystems[0] + sig * self.__drives[0]
