"""Class definition of a coupling optimizable model."""

from typing import override

import jax
import jax.numpy as jnp

from paraqeet.differentiable import Differentiable
from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Array, Quantity

jax.config.update("jax_enable_x64", True)


class Coupling(Optimizable, Differentiable):
    r"""Represents a coupling term between two subsystems. Denoting by O
    the coupling operator and g = \|g\| exp(i phi) the coupling coefficient,
    if add_hermitian = False this adds a term

    \|g\| exp(i phi) * O

    while if True it adds a term

    \|g\| exp(i phi) * O + h.c.
    """

    def __init__(
        self,
        coupling_op: Array,
        g_abs: Quantity,
        g_phase: Quantity = Quantity(0.0, 0.0, 2 * jnp.pi),
        add_hermitian: bool = False,
    ):
        """
        Args:
            coupling_op: The coupling operator. It needs to match the dimension
                of the composite system it is associated with.
            g_abs: Absolute value of the coupling coefficient.
            g_phase: Phase of the coupling coefficient.
            add_hermitian: Whether the Hermitian conjugate of the coupling is
                added or not.
        """
        self.coupling_op = coupling_op
        self.g_abs = g_abs
        self.g_phase = g_phase
        self.add_hermitian = add_hermitian

    @property
    def g_coefficient(self) -> Array:
        """Compute the coupling coefficient.

        Returns:
            The coupling coefficient which is complex in general.
        """
        return self.g_abs.get_value() * jnp.exp(1j * self.g_phase.get_value())

    @override
    def get_parameters(self) -> list[Quantity]:
        return [self.g_abs, self.g_phase]

    @override
    def get_value(self, times: Array) -> Array:
        g_array = self.g_coefficient * jnp.ones((*times.shape, 1, 1))
        coupling_value = g_array * self.coupling_op
        coupling_value += jnp.where(self.add_hermitian, jnp.conjugate(g_array) * self.coupling_op.conj().T, 0.0)
        return coupling_value

    @override
    def get_gradient(self, times: Array) -> Array:
        grad = jnp.empty((*times.shape, 0, *self.coupling_op.shape))
        if self._is_optimized(self.g_abs):
            coeff = jnp.exp(1j * self.g_phase.get_value())
            derivative = coeff * self.coupling_op
            derivative += jnp.where(self.add_hermitian, jnp.conjugate(coeff) * self.coupling_op.conj().T, 0.0)
            grad = jnp.append(grad, derivative * jnp.ones((*times.shape, 1, 1, 1)), axis=1)
        if self._is_optimized(self.g_phase):
            coeff = 1j * jnp.exp(1j * self.g_phase.get_value()) * self.g_abs.get_value()
            derivative = coeff * self.coupling_op
            derivative += jnp.where(self.add_hermitian, jnp.conjugate(coeff) * self.coupling_op.conj().T, 0.0)
            grad = jnp.append(grad, derivative * jnp.ones((*times.shape, 1, 1, 1)), axis=1)
        return grad
