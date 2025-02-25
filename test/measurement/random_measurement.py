"""Class definition of a random measurement model for testing."""

import numpy as np

from cthree.quantity import Quantity
from cthree.measurement.measurement import Measurement
from cthree.propagation.propagation import Propagation


class RandomMeasurement(Measurement):
    """Mock class that returns a random measurement value between 0 and 1.

    Parameters
    ----------
    propagation : cthree.propagation.propagation
        Abstract base class for any implementation
        that can solve the equation of motion.
    times : numpy.ndarray
        One-dimensional vector of timestamps.
    """

    __propagation: Propagation

    def __init__(self, propagation: Propagation, times: np.ndarray):
        super().__init__(times=times)
        self.__propagation = propagation

    def get_parameters(self) -> list[Quantity]:
        """Get parameters of the system.

        Returns
        -------
        list[cthree.quantity]
            The list of parameters of the system.

        """
        return []

    def measure(self) -> np.ndarray:
        """Return the result of measurement.

        Returns
        -------
        numpy.ndarray
            The result of the measurement.

        """
        return np.random.random()
