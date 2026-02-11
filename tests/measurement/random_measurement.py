"""Class definition of a random measurement model for testing."""

import numpy as np

from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array


class RandomMeasurement(NormalizableMeasurement):
    """Mock class that returns a random measurement value between 0 and 1.

    Parameters
    ----------
    propagation : Propagation
        Abstract base class for any implementation
        that can solve the equation of motion.
    times : Array
        One-dimensional vector of timestamps.
    """

    _propagation: Propagation

    def __init__(self, propagation: Propagation):
        self._propagation = propagation

    # TODO: Check the implementation method measure
    def measure(self, times: Array) -> Array | float:
        return self.calculate_normalized_scalar(times=times)

    def calculate_normalized_scalar(self, times: Array | float) -> float:
        """Return the result of measurement.

        Returns
        -------
        Array
            The result of the measurement.

        """
        return float(np.random.random())
