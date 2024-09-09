from typing import List

import numpy as np
import jax.numpy as jnp

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

    def _computeMatrix(self, a: jnp.ndarray) -> jnp.ndarray:
        """
        Returns the operator for the longitudinal or transverse drive.
        """
        return (
            (jnp.conjugate(a.T) @ a)
            if self.__isLongitudinal
            else (jnp.conjugate(a.T) + a)
        )

    def getMatrixOneTime(self, a: jnp.ndarray, t: np.ndarray) -> jnp.ndarray:
        """
        Fetches the coefficient from the drive drive and transforms it into the correct shape for the Hamiltonian.
        """

        signal = self.__generator.generateSignal(t)
        matrix = self._computeMatrix(a)
        return signal * matrix

    def gradientOneTime(self, a: jnp.ndarray, t: float) -> jnp.ndarray:
        """
        Fetches the gradient from the drive and transforms it into the correct shape for the Hamiltonian.
        """
        signalGrad = self.__generator.generateSignalGradient(
            jnp.array(t, ndmin=1)
        ).reshape((-1, 1, 1))
        matrix = self._repeat(self._computeMatrix(a), signalGrad.shape[0])
        return signalGrad * matrix
