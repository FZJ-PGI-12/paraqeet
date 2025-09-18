"""Class definition for a mock system that always returns the same value."""

import jax.numpy as jnp
from paraqeet.quantity import Array

from paraqeet.quantity import Quantity
from paraqeet.measurement.measurement import Measurement
from paraqeet.propagation.propagation import Propagation


class ConstantMeasurement(Measurement):
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
        times: Array = jnp.array(0.0),
    ):
        super().__init__(times=times)
        self.__propagation = propagation
        self.__value = value

    def get_parameters(self) -> list[Quantity]:
        """Get the system parameters.

        Parameters
        ----------
        list[Quantity]
            List of parameters of the system.

        """
        return []

    def measure(self) -> Array:
        """Get the measurement value.

        Returns
        -------
        Array
            The value of the measurement.

        """
        return self.__value

    def measure_with_gradient(self) -> tuple[float, Array]:
        """Get measurement value and gradient"""
        grad = jnp.array([self.__value, 0.0])
        return self.__value, grad
