"""Class definition of a random measurement model for testing."""

import numpy as np

from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array


class RandomMeasurement(NormalizableMeasurement):
    """Mock class that returns a random measurement value between 0 and 1."""

    def __init__(self, propagation: Propagation):
        """
        Args:
            propagation: Abstract base class for any implementation
                that can solve the equation of motion.
        """
        self._propagation = propagation

    def get_value(self, times: Array) -> Array | float:
        return self.calculate_normalized_scalar(times=times)

    def calculate_normalized_scalar(self, times: Array | float) -> float:
        """Return the result of measurement.

        Returns:
            The result of the measurement.
        """
        return float(np.random.random())
