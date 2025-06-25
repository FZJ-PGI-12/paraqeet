"""Class definition of the Transmon Hamiltonian model."""

import jax.numpy as jnp
from cthree.quantity import Array

from cthree.model.drive import Drive
from cthree.model.hamiltonian import Hamiltonian
from cthree.quantity import Quantity

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
    __annihilation_op: Array
    __numOp: Array
    __anharmonic_term: Array

    def __init__(
        self,
        dimension: int,
        frequency: Quantity,
        anharmonicity: Quantity,
        drives: list[Drive] | None = None,
    ):
        super().__init__(drives=drives)
        self.__dimension = dimension
        self.__frequency = frequency
        self.__anharmonicity = anharmonicity
        self.__annihilation_op = jnp.sqrt(jnp.diag(jnp.arange(1, dimension, dtype=jnp.float64), k=1))
        self.__numOp = self.__annihilation_op.T @ self.__annihilation_op
        self.__anharmonic_term = 0.5 * self.__numOp @ (self.__numOp - jnp.eye(self.__dimension))

    def dimension(self) -> int:
        """Get the dimension of the Transmon system."""
        return self.__dimension

    @property
    def frequency(self) -> Quantity:
        """Get the frequency of the Transmon system."""
        return self.__frequency

    @frequency.setter
    def frequency(self, frequency: Quantity) -> None:
        """Set the frequency of the Transmon system."""
        self.__frequency = frequency

    @property
    def anharmonicity(self) -> Quantity:
        """Get the anharmonicity of the Transmon system."""
        return self.__anharmonicity

    @anharmonicity.setter
    def anharmonicity(self, anharmonicity: Quantity) -> None:
        """Set the anharmonicity of the Transmon system."""
        self.__anharmonicity = anharmonicity

    def get_parameters(self) -> list[Quantity]:
        """Get parameters of the model.

        Returns
        -------
        List[cthree.Quantity]
            Returns the list of parameters of the system.

        """
        return self._get_drive_parameters() + [
            self.__frequency,
            self.__anharmonicity,
        ]

    def get_matrix_one_time(self, t: Array) -> Array:
        """Get the drive matrix.

        Parameters
        ----------
        t : chtree.quantity.Array
            Vector of time samples.

        Returns
        -------
        chtree.quantity.Array
            The repeated drive matrix.

        """
        H = self.__frequency.get_value() * self.__numOp + self.__anharmonicity.get_value() * self.__anharmonic_term
        return H + self._get_drive_matrix_one_time(self.__annihilation_op, t)

    def gradient_one_time(self, t: Array) -> Array:
        """Get the gradient of the drive.

        Parameters
        ----------
        t : float
            Single time stamp.

        Returns
        -------
        chtree.quantity.Array
            Returns the gradients of the drive.

        """
        # Fetch the gradient of the drive
        gradients = self._get_drive_gradients_one_time(self.__annihilation_op, t)

        # Combine with the derivatives wrt the frequency and anharmonicity
        grads_list = []
        if self._is_optimised(self.__frequency):
            grads_list.append(self.__numOp)
        if self._is_optimised(self.__anharmonicity):
            grads_list.append(self.__anharmonic_term)
        grads = jnp.stack(grads_list, axis=0) if len(grads_list) > 0 else jnp.empty((0,) + self.__numOp.shape)
        gradients = jnp.append(gradients, grads, axis=0)

        return gradients
