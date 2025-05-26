"""Class definition of the Square Difference Goal model."""

import numpy as np
import jax.numpy as jnp

from cthree.measurement.measurement import Measurement
from cthree.exceptions import ConfigurationException
import itertools


class SquareDifferenceGoal(Measurement):
    """Combine multiple measurements into a single goal function,
    that computes the square of their differences, i.e.,

    \sum_{all pairs (a, b)} (meas_a - meas_b)^2

    Parameters
    ----------
    measurements : list[cthree.measurement.Measurement]
        List of measurements.

    """

    __measurements: list[Measurement]

    def __init__(self, measurements: list[Measurement]):
        super().__init__(None)
        self.__measurements = measurements

    def measure(self) -> np.ndarray:
        """Sum of squared differences

        Returns
        -------
        numpy.ndarray
            Returns the sum of square differences.

        """
        measurements = [m.measure() for m in self.__measurements]
        sumSquareDiff = 0
        
        for meas_a, meas_b in itertools.combinations(measurements, 2):
            sumSquareDiff += (meas_a - meas_b) ** 2
        return sumSquareDiff

    def measure_normalised(self) -> np.ndarray:
        """Sum of squared differences from normalized measurements.

        Returns
        -------
        numpy.ndarray
            Returns the normalized sum of square differences

        """
        measurements = [m.measure_normalised() for m in self.__measurements]
        sumSquareDiff = 0
        for meas_a, meas_b in itertools.combinations(measurements, 2):
            sumSquareDiff += (meas_a - meas_b) ** 2
        return float(sumSquareDiff)

    def measure_with_gradient(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sum of weighted measurements from gradient-ized measurements.

        Returns
        -------
        jax.numpy.ndarray
            Returns the sum of square differences wrt to gradients.
        jax.numpy.ndarray
            Returns gradient of the sum of square difference cost function

        """
        measurements = [m.measure_with_gradient() for m in self.__measurements]
        sumSquareDiff = jnp.array(0)
        grads = jnp.zeros_like(measurements[0][1])

        for meas_a, meas_b in itertools.combinations(measurements, 2):
            sumSquareDiff += (meas_a[0] - meas_b[0]) ** 2
            grads = 2*(meas_a[0] - meas_b[0])*(meas_a[1] - meas_b[1])

        return sumSquareDiff, grads