from typing import List

import jax.numpy as jnp

from cthree.Quantity import Quantity
from cthree.model.Drive import Drive
from cthree.model.Hamiltonian import Hamiltonian

import jax

jax.config.update("jax_enable_x64", True)


class Transmon(Hamiltonian):
    """
    Hamiltonian of an anharmonic oscillator. Optimisable parameters are the ground frequency and the anharmonicity.
    """

    __dimension: int
    __frequency: Quantity
    __anharmonicity: Quantity
    __annihilationOp: jnp.ndarray
    __numOp: jnp.ndarray
    __anharmonicTerm: jnp.ndarray

    def __init__(
        self,
        dimension: int,
        frequency: Quantity,
        anharmonicity: Quantity,
        drives: List[Drive] = None,
    ):
        super().__init__(drives=drives)
        self.__dimension = dimension
        self.__frequency = frequency
        self.__anharmonicity = anharmonicity
        self.__annihilationOp = jnp.sqrt(
            jnp.diag(jnp.arange(1, dimension, dtype=jnp.float64), k=1)
        )
        self.__numOp = self.__annihilationOp.T @ self.__annihilationOp
        self.__anharmonicTerm = (
            0.5 * self.__numOp @ (self.__numOp - jnp.eye(self.__dimension))
        )

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
        return self._getDriveParameters() + [self.__frequency, self.__anharmonicity]

    def getMatrixOneTime(self, t: jnp.ndarray) -> jnp.ndarray:
        H = (
            self.__frequency.getValue() * self.__numOp
            + self.__anharmonicity.getValue() * self.__anharmonicTerm
        )
        return H + self._getDriveMatrixOneTime(self.__annihilationOp, t)

    def gradientOneTime(self, t: float) -> jnp.ndarray:
        # Fetch the gradient of the drive
        gradients = self._getDriveGradientsOneTime(self.__annihilationOp, t)

        # Combine with the derivatives wrt the frequency and anharmonicity
        grads = []
        if self._isOptimised(self.__frequency):
            grads.append(self.__numOp)
        if self._isOptimised(self.__anharmonicity):
            grads.append(self.__anharmonicTerm)
        grads = (
            jnp.stack(grads, axis=0)
            if len(grads) > 0
            else jnp.empty((0,) + self.__numOp.shape)
        )
        gradients = jnp.append(gradients, grads, axis=0)

        return gradients
