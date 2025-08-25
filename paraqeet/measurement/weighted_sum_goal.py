"""Class definition of the Weighted Sum Goal model."""

import jax.numpy as jnp
import numpy as np
from paraqeet.quantity import Array

from paraqeet.exceptions import ConfigurationException
from paraqeet.measurement.measurement import Measurement
from paraqeet.quantity import Quantity
import itertools


class WeightedSumGoal(Measurement):
    """Combine multiple measurements into a single goal function.

    Parameters
    ----------
    measurements : List[Measurement]
        List of measurements.
    weights : Array
        List of weights.

    Raises
    ------
    ConfigurationException
        If number of measurements and weights are incompatible.
    UserWarning
        If the given weights are not normalized.

    """

    __measurements: list[Measurement]
    __weights: Array

    def __init__(self, measurements: list[Measurement], weights: Array, weight_sum_of_squares: float | None = None):
        super().__init__(jnp.array(0.0))
        self.__measurements = measurements
        self.__weights = weights
        self.__weight_sum_of_squares = weight_sum_of_squares
        if len(measurements) != len(weights):
            raise ConfigurationException(
                f"Incompatible number of measurements {len(measurements)}" " and weights {len(weights)}"
            )
        if weight_sum_of_squares is not None:
            if not np.isclose(sum(weights) + weight_sum_of_squares, 1.0):
                raise UserWarning("Total supplied weights are not normalized.")
        else:
            if not np.isclose(sum(weights), 1.0):
                raise UserWarning("Supplied weights are not normalized.")

    def get_parameters(self) -> list[Quantity]:
        """Returns an empty list."""
        return []

    def measure(self) -> Array:
        """Sum of plain weighted measurements.

        Returns
        -------
        Array
            Returns the plain weighted sum.

        """
        measurements = [m.measure() for m in self.__measurements]
        sum_meas = 0.0
        for ii, w in enumerate(self.__weights):
            sum_meas += w * measurements[ii]
        if self.__weight_sum_of_squares is not None:
            sum_square_diff = 0.0
            for meas_a, meas_b in itertools.combinations(measurements, 2):
                sum_square_diff += (meas_a - meas_b) ** 2
            sum_meas += self.__weight_sum_of_squares * sum_square_diff
        return sum_meas

    def measure_normalised_scalar(self) -> float:
        """Sum of weighted measurements from normalized measurements.

        Returns
        -------
        Array
            Returns the normalized weighted sum.

        """
        measurements = [m.measure_normalised_scalar() for m in self.__measurements]
        sum_meas = 0.0
        for ii, w in enumerate(self.__weights):
            sum_meas += w * measurements[ii]
        if self.__weight_sum_of_squares is not None:
            sum_square_diff = 0.0
            for meas_a, meas_b in itertools.combinations(measurements, 2):
                sum_square_diff += (meas_a - meas_b) ** 2
            sum_meas += self.__weight_sum_of_squares * sum_square_diff
        return float(sum_meas)

    def measure_with_gradient(self) -> tuple[float, Array]:
        """Sum of weighted measurements from gradient-ized measurements.

        Returns
        -------
        paraqeet.quantity.Array
            Returns the weighted sum wrt to gradients.
        paraqeet.quantity.Array
            Returns the sum of gradients.

        """
        measurements = [m.measure_with_gradient() for m in self.__measurements]
        sum_meas = jnp.array(0)
        sum_grads = jnp.zeros_like(measurements[0][1])
        for ii, w in enumerate(self.__weights):
            sum_meas += w * measurements[ii][0]
            sum_grads += w * measurements[ii][1]
        if self.__weight_sum_of_squares is not None:
            sum_square_diff = jnp.array(0)
            grads_diff = jnp.zeros_like(measurements[0][1])

            for meas_a, meas_b in itertools.combinations(measurements, 2):
                sum_square_diff += (meas_a[0] - meas_b[0]) ** 2
                grads_diff += 2 * (meas_a[0] - meas_b[0]) * (meas_a[1] - meas_b[1])
            sum_meas += self.__weight_sum_of_squares * sum_square_diff
            sum_grads += self.__weight_sum_of_squares * grads_diff
        return sum_meas, sum_grads
