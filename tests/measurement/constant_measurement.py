"""Class definition for a mock system that always returns the same value."""

from typing import override

import jax.numpy as jnp

from paraqeet.differentiable import Differentiable
from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array


class ConstantMeasurement(NormalizableMeasurement, Differentiable):
    """Mock implementation that always returns the same value.

    Parameters
    ----------
    propagation : Propagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    value : float, default=1.0
        Value of the measurement.
    times : float | None, optional
        Time variable value.
    """

    _propagation: Propagation
    _value: Array

    def __init__(
        self,
        propagation: Propagation,
        value: Array = jnp.array(1.0),
    ):
        self._propagation = propagation
        self._value = value

    @override
    def get_value(self, times: Array) -> Array:
        return self._value

    @override
    def measure(self, times: Array) -> Array:
        return self.get_value(times)

    @override
    def calculate_normalized_scalar(self, times: Array) -> Array:
        return self.get_value(times)

    def get_gradient(self, times: Array) -> Array:
        """Get measurement value and gradient"""
        grad = jnp.array([self._value, 0.0])
        return grad
