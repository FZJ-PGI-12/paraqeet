"""Class definition of a Drive optimizable model."""

from typing import override

import jax
import jax.numpy as jnp
from jax import vmap

from paraqeet.differentiable import Differentiable
from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Array, Float, Quantity
from paraqeet.signal.generator import Generator


# TODO: Is Drive a Differentiable object? If so, it should inherit from Differentiable at least
# for the purpose of clarity and consistency. Would it make sense to add a default
# implementation of the abstract Differentiable method get_value_and_gradient?
# If not, we should rename the methods to e.g. get_hamiltonian_gradient to avoid confusion.
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
        if added or not

    """

    def __init__(self, drive_op: Array, generator: Generator, add_hermitian: bool = False) -> None:
        self.drive_op = drive_op
        self.generator = generator
        self.add_hermitian = add_hermitian

    def get_parameters(self) -> list[Quantity]:
        """Get a list of parameters of the system.

        Returns
        -------
        list[Quantity]
            List of optimizable parameters of the system.

        """
        return self.generator.get_parameters()

    def get_value_at_timestep(self, t: float) -> Array:
        """Get the one-time matrix of the system.

        Fetches the coefficient from the drive and transforms it
        into the correct shape for the Hamiltonian.

        Parameters
        ----------
        t: float
            Time.

        Returns
        -------
        Array
            Returns the shape-shifted coefficient from the drive.

        """
        # TODO: generator.get_value expects an array, even for one time point.
        # Is the naming of the method correct then?
        signal = self.generator.get_value(t)
        drive = (
            signal * self.drive_op
            + jax.lax.cond(self.add_hermitian, lambda: 1.0, lambda: 0.0)
            * jnp.conjugate(signal)
            * self.drive_op.conj().T
        )
        return drive

    def get_value(self, times: Array) -> Array:
        """Return the matrix representation of the drive.

        Parameters
        ----------
        times: Array
            Vector of time samples.

        Returns
        -------
        Array
            Matrix of shape [t, n, n]  with 't' as time and 'n' as the Hilbert
            space dimension.

        """
        # vmap iterates over the times array and returns float. Not caught by mypy.
        drive_value = vmap(self.get_value_at_timestep)(times)  # type: ignore
        return drive_value

    def get_gradient_at_timestep(self, t: float) -> Array:
        """Get the one-time gradient of the system.

        Fetches the gradient from the drive and transforms it into the
        correct shape for the Hamiltonian.

        Parameters
        ----------
        t: Array
            Time.

        Returns
        -------
        Array
            Returns the shape-shifted gradient from the drive.

        """
        _, signal_grad = self.generator.get_value_and_gradient(jnp.array([t]))
        signal_grad = signal_grad.reshape((-1, 1, 1))
        drive_grad = (
            signal_grad * self.drive_op
            + jax.lax.cond(self.add_hermitian, lambda: 1.0, lambda: 0.0)
            + jnp.conjugate(signal_grad) * self.drive_op.conj().T
        )
        return drive_grad

    def get_gradient(self, times: Array) -> Array:
        """Return the gradient of the system.

        Returns the gradient of the matrix representation of the drive Hamiltonian
        with respect to each parameter as a list.

        Parameters
        ----------
        times: Array
            Vector of time samples.

        Returns
        -------
        Array
            Array of shape [t, p, n, n] with 't' as time, 'p' as number of
            parameters and 'n' as the Hilbert space dimension.


        """
        # Ignoring mypy here as vmap makes the array to float
        gradient_value = vmap(self.get_gradient_at_timestep)(times)  #  type: ignore

        return gradient_value

    @override
    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array] | tuple[Float, Array]:
        return self.get_value(times), self.get_gradient(times)
