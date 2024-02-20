from typing import List

import numpy as np
import jax.numpy as jnp

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
    __annihilationOp: jnp.ndarray
    __sigmaZ: np.array

    def __init__(self, frequency: Quantity, drives: List[Drive] = None):
        super().__init__(drives)
        self.__frequency = frequency
        self.__annihilationOp = jnp.array([
            [0.0, 0.0],
            [1.0, 0.0],
        ])
        self.__sigmaZ = np.diag([1.0, -1.0])

    def getFrequency(self) -> Quantity:
        return self.__frequency

    def setFrequency(self, frequency: Quantity) -> None:
        self.__frequency = frequency

    def getParameters(self) -> List[Quantity]:
        return [self.__frequency] + self._getDriveParameters()

    def dimension(self) -> int:
        return 2

    def getMatrix(self, t: jnp.ndarray) -> jnp.ndarray:
        H = 0.5 * self.__frequency.getValue() * self.__sigmaZ
        return self._repeat(H, t.shape[0]) + self._getDriveMatrix(self.__annihilationOp, t)

    def gradient(self, t: jnp.ndarray) -> jnp.ndarray:
        # Fetch the gradient of the drive
        derivatives = self._getDriveGradients(self.__annihilationOp, t)

        # Combine with the derivative wrt the frequency
        if self._isOptimised(self.__frequency):
            H = 0.5 * self.__sigmaZ.reshape((1, 2, 2))
            derivative = self._repeat(H, t.shape[0])
            derivatives = jnp.append(derivatives, derivative, axis=1)

        return derivatives
