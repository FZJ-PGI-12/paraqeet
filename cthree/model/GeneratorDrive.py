from typing import List

import numpy as np

from cthree.Quantity import Quantity
from cthree.model.Drive import Drive
from cthree.signal.Generator import Generator


class GeneratorDrive(Drive):
    """
    Transversal (a^\\dagger a) or longitudinal (a^\\dagger a) drive with a time-dependent scalar coefficient that is
    generator by a Generator object.
    """
    __generator: Generator
    __isLongitudinal: bool

    def __init__(self, generator: Generator, isLongitudinal: bool):
        self.__generator = generator
        self.__isLongitudinal = isLongitudinal

    def getGenerator(self) -> Generator:
        return self.__generator

    def getParameters(self) -> List[Quantity]:
        return self.__generator.getParameters()

    def getMatrix(self, a: np.ndarray, t: np.ndarray) -> np.ndarray:
        signal = self.__generator.generateSignal(t)
        matrix = (np.conjugate(a.T) @ a) if self.__isLongitudinal else (np.conjugate(a.T) + a)
        return signal.reshape((signal.shape[0], 1, 1)) * self._repeatInTime(matrix, t)

    def gradient(self, a: np.ndarray, t: np.ndarray) -> np.ndarray:
        # TODO
        pass