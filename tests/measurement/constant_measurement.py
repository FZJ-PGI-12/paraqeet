"""Class definition for a mock system that always returns the same value."""

import jax.numpy as jnp

from paraqeet.differentiable import Differentiable
from paraqeet.measurement.measurement import Measurement
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array


class ConstantMeasurement(Measurement, Differentiable):
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

    __propagation: Propagation
    __value: Array

    def __init__(
        self,
        propagation: Propagation,
        value: Array = jnp.array(1.0),
    ):
        self.__propagation = propagation
        self.__value = value

    def measure(self, times: Array) -> Array:
        """Get the measurement value.

        Returns
        -------
        Array
            The value of the measurement.

        """
        return self.__value

    def value_and_gradient(self, times: Array) -> tuple[Array, Array] | tuple[float, Array]:
        """Get measurement value and gradient"""
        grad = jnp.array([self.__value, 0.0])
        return self.__value, grad
