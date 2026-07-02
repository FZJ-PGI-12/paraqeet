"""Class definition of a Drive optimizable model."""

from typing import override

import jax.numpy as jnp

from paraqeet.differentiable import Differentiable
from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Array, Quantity
from paraqeet.signal.generator import Generator


class Drive(Optimizable, Differentiable):
    """Represents a time-dependent drive on a system.

    This can for example be a microwave or flux drive.

    Parameters
    ----------
    drive_op: Array
        The drive operator. It needs to match the dimension of the system
        it is associated with.
    generator : Generator
        Signal generator.
    add_hermitian: bool=False
        A boolean that determines whether the Hermitian conjugate of the drive
        is added or not

    """

    def __init__(self, drive_op: Array, generator: Generator, add_hermitian: bool = False) -> None:
        self.drive_op = drive_op
        self.generator = generator
        self.add_hermitian = add_hermitian

    @override
    def get_parameters(self) -> list[Quantity]:
        """Get a list of parameters of the system.

        Returns
        -------
        list[Quantity]
            List of optimizable parameters of the system.

        """
        return self.generator.get_parameters()

    @override
    def get_value(self, times: Array) -> Array:
        """Return the matrix representation of the drive.

        Parameters
        ----------
        times: Array
            Array of times.

        Returns
        -------
        Array
            Matrix of shape [n_times, n, n]  with n_times as the number of times
            and 'n' as the Hilbert space dimension.
        """
        signal = self.generator.get_value(times)
        signal = signal.reshape(*signal.shape, 1, 1)
        drive_value = signal * self.drive_op
        drive_value += jnp.where(self.add_hermitian, jnp.conjugate(signal) * self.drive_op.conj().T, 0.0)
        return drive_value

    @override
    def get_gradient(self, times: Array) -> Array:
        """Return the gradient of the system.

        Returns the gradient of the matrix representation of the drive Hamiltonian
        with respect to each parameter as a list.

        Parameters
        ----------
        times: Array
            Array of times.

        Returns
        -------
        Array
            Array of shape [n_times, n_params, n, n] with n_times as the number of times,
            n_params as number of parameters and 'n' as the Hilbert space dimension.
        """
        signal_grad = self.generator.get_gradient(times)
        signal_grad = signal_grad.reshape(*signal_grad.shape, 1, 1)
        drive_grad = self.drive_op * signal_grad
        drive_grad += jnp.where(self.add_hermitian, jnp.conjugate(signal_grad) * self.drive_op.conj().T, 0.0)
        return drive_grad
