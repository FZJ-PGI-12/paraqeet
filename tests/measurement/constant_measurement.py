"""Class definition for a mock system that always returns the same value."""

import jax.numpy as jnp

from paraqeet.differentiable import Differentiable
from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array


class ConstantMeasurement(NormalizableMeasurement, Differentiable):
    """Mock implementation that always returns the same value."""

    def __init__(
        self,
        propagation: Propagation,
        value: Array = jnp.array(1.0),
    ):
        """
        Args:
            propagation: Abstract base class for any implementation that can solve
                the equation of motion.
            value: Value of the measurement. Defaults to 1.0.
        """
        self._propagation = propagation
        self._value = value

    def get_value(self, times: Array) -> Array:
        return self._value

    def measure(self, times: Array) -> Array:
        return self.get_value(times)

    def calculate_normalized_scalar(self, times: Array) -> Array:
        return self.get_value(times)

    def get_gradient(self, times: Array) -> Array:
        """Get measurement value and gradient"""
        grad = jnp.array([self._value, 0.0])
        return grad
