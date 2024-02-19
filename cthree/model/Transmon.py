from typing import List

import jax.numpy as jnp

from cthree.Quantity import Quantity
from cthree.model.Drive import Drive
from cthree.model.Hamiltonian import Hamiltonian


class Transmon(Hamiltonian):
    """
    Hamiltonian of an anharmonic oscillator. Optimisable parameters are the ground frequency and the anharmonicity.
    """
    __dimension: int
    __frequency: Quantity
    __anharmonicity: Quantity
    __annihilationOp: jnp.ndarray
    __numOp: jnp.ndarray

    def __init__(self, dimension: int, frequency: Quantity, anharmonicity: Quantity, drives: List[Drive] = None):
        super().__init__(drives=drives)
        self.__dimension = dimension
        self.__frequency = frequency
        self.__anharmonicity = anharmonicity
        self.__annihilationOp = jnp.sqrt(jnp.diag(jnp.arange(1, dimension, dtype=jnp.float64), k=1))
        self.__numOp = self.__annihilationOp.T @ self.__annihilationOp

    def dimension(self) -> int:
        return self.__dimension

    def getFrequency(self) -> Quantity:
        return self.__frequency

    def setFrequency(self, frequency: Quantity) -> None:
        self.__frequency = frequency

    def getAnharmonicity(self) -> Quantity:
        return self.__anharmonicity

    def setAnharmonicity(self, anharmonicity: Quantity) -> None:
        self.__anharmonicity = anharmonicity

    def getParameters(self) -> List[Quantity]:
        return [self.__frequency, self.__anharmonicity] + self._getDriveParameters()

    def __constructOperators(self) -> List[jnp.ndarray]:
        """
        Returns the operators for the two terms of the matrix or gradient without coefficients.
        """
        return [
            self.__numOp,
            0.5 * self.__numOp @ (self.__numOp - jnp.eye(self.__dimension))
        ]

    def getMatrix(self, t: jnp.ndarray) -> jnp.ndarray:
        ops = self.__constructOperators()
        H = self.__frequency.getValue() * ops[0] + self.__anharmonicity.getValue() * ops[1]
        return self._repeat(H, t.shape[0]) + self._getDriveMatrix(self.__annihilationOp, t)

    def gradient(self, t: jnp.ndarray) -> jnp.ndarray:
        # Derivatives wrt to the frequency and the anharmonicity
        grads = jnp.stack(self.__constructOperators(), axis=0)
        grads = self._repeat(grads, t.shape[0])

        # Combine with the derivatives of the drives
        driveGradients = self._getDriveGradients(self.__annihilationOp, t)
        allGrads = jnp.append(driveGradients, grads, axis=1)
        return allGrads
