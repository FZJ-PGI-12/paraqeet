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
    """

    def __init__(self, drive_op: Array, generator: Generator, add_hermitian: bool = False) -> None:
        """
        Args:
            drive_op: The drive operator. It needs to match the dimension of the
                system it is associated with.
            generator: Signal generator.
            add_hermitian: Whether the Hermitian conjugate of the drive is added
                or not.
        """
        self.drive_op = drive_op
        self.generator = generator
        self.add_hermitian = add_hermitian

    @override
    def get_parameters(self) -> list[Quantity]:
        return self.generator.get_parameters()

    @override
    def get_value(self, times: Array) -> Array:
        signal = self.generator.get_value(times)
        signal = signal.reshape(*signal.shape, 1, 1)
        drive_value = signal * self.drive_op
        drive_value += jnp.where(self.add_hermitian, jnp.conjugate(signal) * self.drive_op.conj().T, 0.0)
        return drive_value

    @override
    def get_gradient(self, times: Array) -> Array:
        signal_grad = self.generator.get_gradient(times)
        signal_grad = signal_grad.reshape(*signal_grad.shape, 1, 1)
        drive_grad = self.drive_op * signal_grad
        drive_grad += jnp.where(self.add_hermitian, jnp.conjugate(signal_grad) * self.drive_op.conj().T, 0.0)
        return drive_grad
