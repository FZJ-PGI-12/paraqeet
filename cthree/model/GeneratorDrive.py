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

    def _computeMatrix(self, a: np.ndarray) -> np.ndarray:
        return (np.conjugate(a.T) @ a) if self.__isLongitudinal else (np.conjugate(a.T) + a)

    def getMatrix(self, a: np.ndarray, t: np.ndarray) -> np.ndarray:
        signal = self.__generator.generateSignal(t)
        matrix = self._computeMatrix(a)
        return signal.reshape((signal.shape[0], 1, 1)) * self._repeat(matrix, t.shape[0])

    def gradient(self, a: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Fetches the gradient from the drive and transforms it into the correct shape for the Hamiltonian.
        """
        signalGrad = self.__generator.generateSignalGradient(t) # (t, p)
        matrix = self._computeMatrix(a)
        matrix = self._repeat(self._repeat(matrix, signalGrad.shape[1]), t.shape[0])
        return signalGrad.reshape(signalGrad.shape + (1, 1)) * matrix
