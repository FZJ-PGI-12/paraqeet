"""Class definition for a mock system that always returns the same value."""

import jax.numpy as jnp
from paraQeet.quantity import Array

from paraQeet.quantity import Quantity
from paraQeet.measurement.measurement import Measurement
from paraQeet.propagation.propagation import Propagation


class ConstantMeasurement(Measurement):
    """Mock implementation that always returns the same value.

    Parameters
    ----------
    propagation : paraqeet.propagation.propagation
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
        list[paraqeet.Quantity]
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
