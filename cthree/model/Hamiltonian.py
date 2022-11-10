from typing import List

import numpy as np

from cthree.generation.Generator import Generator


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

    def __init__(self, subsystems: List, couplings: List, drives: List, generator: Generator):
        self.__subsystems = subsystems
        self.__couplings = couplings
        self.__drives = drives
        self.__generator = generator

    def getMatrix(self) -> np.ndarray:
        self.__generator.generateSignal()
        pass
