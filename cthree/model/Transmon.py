"""Class definition of the Transmon Hamiltonian model."""

import jax.numpy as jnp

from cthree.Quantity import Quantity
from cthree.model.Drive import Drive
from cthree.model.Hamiltonian import Hamiltonian

import jax

jax.config.update("jax_enable_x64", True)


class Transmon(Hamiltonian):
    """Hamiltonian of an anharmonic oscillator.

    Optimisable parameters are the ground frequency and the anharmonicity.

    Parameters
    ----------
    dimension : int
        Dimension of the anharmonic oscillator.
    frequency : cthree.model.Quantity
        Frequency of the anharmonic oscillator.
    anharmonicity : cthree.model.Quantity
        Anharmonicity of the oscillator.
    drives : List[cthree.model.Drive], optional
        List of time-dependent drives of the subsystem.

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
        drives: list[Drive] = None,
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
        """Get the dimension of the Transmon system."""
        return self.__dimension

    def getFrequency(self) -> Quantity:
        """Get the frequency of the Transmon system."""
        return self.__frequency

    def setFrequency(self, frequency: Quantity) -> None:
        """Set the frequency of the Transmon system."""
        self.__frequency = frequency

    def getAnharmonicity(self) -> Quantity:
        """Get the anharmonicity of the Transmon system."""
        return self.__anharmonicity

    def setAnharmonicity(self, anharmonicity: Quantity) -> None:
        """Set the anharmonicity of the Transmon system."""
        self.__anharmonicity = anharmonicity

    def getParameters(self) -> list[Quantity]:
        """Get parameters of the model.

        Returns
        -------
        List[cthree.Quantity]
            Returns the list of parameters of the system.

        """
        return self._getDriveParameters() + [
            self.__frequency,
            self.__anharmonicity,
        ]

    def getMatrixOneTime(self, t: jnp.ndarray) -> jnp.ndarray:
        """Get the drive matrix.

        Parameters
        ----------
        t : jax.numpy.ndarray
            Vector of time samples.

        Returns
        -------
        jax.numpy.ndarray
            The repeated drive matrix.

        """
        H = (
            self.__frequency.getValue() * self.__numOp
            + self.__anharmonicity.getValue() * self.__anharmonicTerm
        )
        return H + self._getDriveMatrixOneTime(self.__annihilationOp, t)

    def gradientOneTime(self, t: float) -> jnp.ndarray:
        """Get the gradient of the drive.

        Parameters
        ----------
        t : float
            Single time stamp.

        Returns
        -------
        jax.numpy.ndarray
            Returns the gradients of the drive.

        """
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
