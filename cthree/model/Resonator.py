from typing import List

import jax.numpy as jnp

from cthree.Quantity import Quantity
from cthree.model.Drive import Drive
from cthree.model.Hamiltonian import Hamiltonian


class Resonator(Hamiltonian):
    """
    Hamiltonian of a harmonic oscillator. The only optimisable parameter is the frequency.
    """
    __dimension: int
    __frequency: Quantity
    __annihilationOp: jnp.ndarray
    __numOp: jnp.ndarray

    def __init__(self, dimension: int, frequency: Quantity, drives: List[Drive] = None):
        super().__init__(drives=drives)
        self.__dimension = dimension
        self.__frequency = frequency
        self.__annihilationOp = jnp.sqrt(jnp.diag(jnp.arange(1, dimension, dtype=jnp.float64), k=1))
        self.__numOp = self.__annihilationOp.T @ self.__annihilationOp

    def dimension(self):
        return self.__dimension

    def getFrequency(self) -> Quantity:
        return self.__frequency

    def setFrequency(self, frequency: Quantity) -> None:
        self.__frequency = frequency

    def getParameters(self) -> List[Quantity]:
        return [self.__frequency] + self._getDriveParameters()

    def getMatrix(self, t: jnp.ndarray) -> jnp.ndarray:
        H = self.__frequency * self.__numOp
        return self._repeat(H, t.shape[0]) + self._getDriveMatrix(self.__annihilationOp, t)

    def gradient(self, t: jnp.ndarray) -> jnp.ndarray:
        # Fetch the gradient of the drive
        derivatives = self._getDriveGradients(self.__annihilationOp, t)

        # Combine with the derivative wrt the frequency
        if self._isOptimised(self.__frequency):
            grad = self.__numOp.reshape((1,) + self.__numOp.shape)
            grad = self._repeat(grad, t.shape[0])
            derivatives = jnp.append(derivatives, grad, axis=1)

        return derivatives
