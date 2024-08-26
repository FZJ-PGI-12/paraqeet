from typing import List, Tuple

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
    __anharmonicTerm: jnp.ndarray
    __t1: Quantity
    __temp: Quantity
    __t2star: Quantity

    def __init__(
        self,
        dimension: int,
        frequency: Quantity,
        anharmonicity: Quantity,
        drives: List[Drive] = None,
        t1: Quantity = None,
        temp: Quantity = None,
        t2star: Quantity = None,
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
        self.__t1 = t1
        self.__temp = temp
        self.__t2star = t2star

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

    def getDecayRates(self) -> List[float]:
        if (self.__t1 is None) or (self.__t2star is None) or (self.__temp is None):
            raise Exception(
                "Specify values of T1, T2star and Temp for Open system simulations."
            )

        gamma = 1 / self.__t1.getValue()
        gammaT2star = 0.5 / self.__t2star.getValue()

        hbar_over_kb = 7.638232582257738e-12
        beta = hbar_over_kb / (self.__temp.getValue())
        # TODO - This would have anharmonicity term too. Add that.
        nbar = jnp.exp(-beta * self.__frequency.getValue())  # TODO - Check this part
        gammaTemp = gamma * nbar  # TODO - Check this part
        gammaT1 = gamma * (nbar + 1)  # TODO - Check this part
        return [gammaT1, gammaTemp, gammaT2star]

    def getCollapseOps(self) -> List[Tuple[float, jnp.ndarray]]:
        """
        Return a list tuples of decay rates and collapse operators for each subsystem.

        Returns:
            List[Tuple[float, np.ndarray]]: List of collapse operators
        """
        gammaT1, gammaTemp, gammaT2star = self.getDecayRates()
        col_t1 = self.__annihilationOp
        col_temp = self.__annihilationOp.T
        col_t2star = 2 * self.__numOp
        return [(gammaT1, col_t1), (gammaTemp, col_temp), (gammaT2star, col_t2star)]
