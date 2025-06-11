"""Class definition of the Weighted Sum Goal model."""

import numpy as np
import jax.numpy as jnp

from cthree.measurement.measurement import Measurement
from cthree.exceptions import ConfigurationException
import itertools


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
    __weights: np.ndarray

    def __init__(self, measurements: list[Measurement], weights: np.ndarray, weight_sum_of_squares: float | None = None):
        super().__init__(None)
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


    def measure(self) -> np.ndarray:
        """Sum of plain weighted measurements.

        Returns
        -------
        numpy.ndarray
            Returns the plain weighted sum.

        """
        measurements = [m.measure() for m in self.__measurements]
        sumMeas = 0
        for ii, w in enumerate(self.__weights):
            sumMeas += w * measurements[ii]
        if self.__weight_sum_of_squares is not None:
            sumSquareDiff = 0
            for meas_a, meas_b in itertools.combinations(measurements, 2):
                sumSquareDiff += (meas_a - meas_b) ** 2
            sumMeas += self.__weight_sum_of_squares*sumSquareDiff
        return sumMeas

    def measure_normalised(self) -> np.ndarray:
        """Sum of weighted measurements from normalized measurements.

        Returns
        -------
        numpy.ndarray
            Returns the normalized weighted sum.

        """
        measurements = [m.measure_normalised() for m in self.__measurements]
        sumMeas = 0
        for ii, w in enumerate(self.__weights):
            sumMeas += w * measurements[ii]
        if self.__weight_sum_of_squares is not None:
            sumSquareDiff = 0
            for meas_a, meas_b in itertools.combinations(measurements, 2):
                sumSquareDiff += (meas_a - meas_b) ** 2
            sumMeas += self.__weight_sum_of_squares*sumSquareDiff
        return float(sumMeas)

    def measure_with_gradient(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sum of weighted measurements from gradient-ized measurements.

        Returns
        -------
        jax.numpy.ndarray
            Returns the weighted sum wrt to gradients.
        jax.numpy.ndarray
            Returns the sum of gradients.

        """
        measurements = [m.measure_with_gradient() for m in self.__measurements]
        sumMeas = jnp.array(0)
        sumGrads = jnp.zeros_like(measurements[0][1])
        for ii, w in enumerate(self.__weights):
            sumMeas += w * measurements[ii][0]
            sumGrads += w * measurements[ii][1]
        if self.__weight_sum_of_squares is not None:
            sumSquareDiff = jnp.array(0)
            gradsDiff = jnp.zeros_like(measurements[0][1])

            for meas_a, meas_b in itertools.combinations(measurements, 2):
                sumSquareDiff += (meas_a[0] - meas_b[0]) ** 2
                gradsDiff += 2*(meas_a[0] - meas_b[0])*(meas_a[1] - meas_b[1])
            sumMeas += self.__weight_sum_of_squares*sumSquareDiff
            sumGrads += self.__weight_sum_of_squares*gradsDiff
        return sumMeas, sumGrads
