"""Class definition of the Resonator Hamiltonian model."""

import jax.numpy as jnp

from cthree.Quantity import Quantity
from cthree.model.Drive import Drive
from cthree.model.Hamiltonian import Hamiltonian

import jax

jax.config.update("jax_enable_x64", True)


class Resonator(Hamiltonian):
    """Hamiltonian of a harmonic oscillator.

    The only optimisable parameter is the frequency.

    Parameters
    ----------
    dimension : int
        Dimension of the harmonic oscillator.
    frequency : cthree.model.Quantity
        Frequency of the harmonic oscillator.
    drives : List[cthree.model.Drive], optional
        List of time-dependent drives of the subsystem.

    """

    __dimension: int
    __frequency: Quantity
    __annihilation_op: jnp.ndarray
    __numOp: jnp.ndarray

    def __init__(self, dimension: int, frequency: Quantity, drives: list[Drive] = None):
        super().__init__(drives=drives)
        self.__dimension = dimension
        self.__frequency = frequency
        self.__annihilation_op = jnp.sqrt(jnp.diag(jnp.arange(1, dimension, dtype=jnp.float64), k=1))
        self.__numOp = self.__annihilation_op.T @ self.__annihilation_op

    def dimension(self):
        """Get the dimension of the resonator."""
        return self.__dimension

    def get_frequency(self) -> Quantity:
        """Get the frequency of the resonator."""
        return self.__frequency

    def set_frequency(self, frequency: Quantity) -> None:
        """Set the frequency of the resonator."""
        self.__frequency = frequency

    def get_parameters(self) -> list[Quantity]:
        """Get parameters of the model.

        Returns
        -------
        List[cthree.Quantity]
            Returns the list of parameters of the system.

        """
        return self._get_drive_parameters() + [self.__frequency]

    def get_matrix_one_time(self, t: float) -> jnp.ndarray:
        """Get the drive matrix.

        Parameters
        ----------
        t : float
            One time stamp.

        Returns
        -------
        jax.numpy.ndarray
            The drive matrix at a single timestamp.

        """
        H = self.__frequency.get_value() * self.__numOp
        return H + self._get_drive_matrix_one_time(self.__annihilation_op, t)

    def gradient_one_time(self, t: float) -> jnp.ndarray:
        """Get the gradient of the drive.

        Parameters
        ----------
        t : float
            One time stamp.

        Returns
        -------
        jax.numpy.ndarray
            Returns the gradients of the drive.

        """
        # Fetch the gradient of the drive
        derivatives = self._get_drive_gradients_one_time(self.__annihilation_op, t)

        # Combine with the derivative wrt the frequency
        if self._is_optimised(self.__frequency):
            grad = self.__numOp.reshape((1,) + self.__numOp.shape)
            derivatives = jnp.append(derivatives, grad, axis=0)

        return derivatives
