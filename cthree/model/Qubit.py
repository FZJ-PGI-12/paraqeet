from typing import List

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
    __drift: jnp.array

    def __init__(self, frequency: Quantity, drives: List[Drive] = None):
        super().__init__(drives)
        self.__frequency = frequency
        self.__annihilationOp = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        )
        self.__drift = 0.5 * jnp.diag(jnp.array([1.0, -1.0]))

    def getFrequency(self) -> Quantity:
        return self.__frequency

    def setFrequency(self, frequency: Quantity) -> None:
        self.__frequency = frequency

    def getParameters(self) -> List[Quantity]:
        return self._getDriveParameters() + [self.__frequency]

    def dimension(self) -> int:
        return 2

    def getMatrixOneTime(self, t: float) -> jnp.ndarray:
        H = self.__frequency.getValue() * self.__drift
        return H + self._getDriveMatrixOneTime(self.__annihilationOp, t)

    def gradientOneTime(self, t: float) -> jnp.ndarray:
        # Fetch the gradient of the drive
        derivatives = self._getDriveGradientsOneTime(self.__annihilationOp, t)

        # Combine with the derivative wrt the frequency
        if self._isOptimised(self.__frequency):
            H = self.__drift.reshape((1, 2, 2))
            derivatives = jnp.append(derivatives, H, axis=0)

        return derivatives
