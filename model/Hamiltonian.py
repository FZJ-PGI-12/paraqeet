from typing import List

import numpy as np

from generation.Generator import Generator


class Hamiltonian:
    """
    Matrix representation of a Hamiltonian.
    Contains subsystems, couplings, and drive lines.
    Takes care of frame transformations.
    """
    __subsystems: List
    __couplings: List
    __drives: List
    __generator: Generator

    def getMatrix(self) -> np.array:
        self.__generator.generateSignal()
        pass
