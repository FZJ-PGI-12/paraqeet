"""Class definition of the Weighted Sum Goal model."""

import jax.numpy as jnp
from jax import Array

from cthree.exceptions import ConfigurationException
from cthree.measurement.measurement import Measurement
from cthree.quantity import Quantity


class WeightedSumGoal(Measurement):
    """Combine multiple measurements into a single goal function.

    Parameters
    ----------
    measurements : List[cthree.measurement.Measurement]
        List of measurements.
    weights : numpy.ndarray
        List of weights.

    Raises
    ------
    cthree.Exceptions.ConfigurationException
        If number of measurements and weights are incompatible.
    UserWarning
        If the given weights are not normalized.

    """

    __measurements: list[Measurement]
    __weights: Array

    def __init__(self, measurements: list[Measurement], weights: Array):
        super().__init__(jnp.array(0.0))
        self.__measurements = measurements
        self.__weights = weights
        if len(measurements) != len(weights):
            raise ConfigurationException(
                f"Incompatible number of measurements {len(measurements)}" " and weights {len(weights)}"
            )
        if not jnp.isclose(sum(weights), 1.0):
            raise UserWarning("Supplied weights are not normalized.")

    def get_parameters(self) -> list[Quantity]:
        """Returns an empty list."""
        return []

    def measure(self) -> Array:
        """Sum of plain weighted measurements.

        Returns
        -------
        numpy.ndarray
            Returns the plain weighted sum.

        """
        measurements = [m.measure() for m in self.__measurements]
        sumMeas = jnp.array(0.0)
        for ii, w in enumerate(self.__weights):
            sumMeas += w * measurements[ii]
        return sumMeas

    def measure_normalised(self) -> float:
        """Sum of weighted measurements from normalized measurements.

        Returns
        -------
        numpy.ndarray
            Returns the normalized weighted sum.

        """
        measurements = [m.measure_normalised_scalar() for m in self.__measurements]
        sumMeas = 0.0
        for ii, w in enumerate(self.__weights):
            sumMeas += w * measurements[ii]
        return sumMeas

    def measure_with_gradient(self) -> tuple[float, Array]:
        """Sum of weighted measurements from gradient-ized measurements.

        Returns
        -------
        jax.Array
            Returns the weighted sum wrt to gradients.
        jax.numpy.ndarray
            Returns the sum of gradients.

        """
        measurements = [m.measure_with_gradient() for m in self.__measurements]
        sumMeas = 0.0
        sumGrads = jnp.zeros_like(measurements[0][1])
        for ii, w in enumerate(self.__weights):
            sumMeas += w * measurements[ii][0]
            sumGrads += w * measurements[ii][1]
        return sumMeas, sumGrads
