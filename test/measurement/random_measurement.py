"""Class definition of a random measurement model for testing."""

import numpy as np
from paraQeet.quantity import Array
from paraQeet.quantity import Quantity
from paraQeet.measurement.measurement import Measurement
from paraQeet.propagation.propagation import Propagation


class RandomMeasurement(Measurement):
    """Mock class that returns a random measurement value between 0 and 1.

    Parameters
    ----------
    propagation : cthree.propagation.propagation
        Abstract base class for any implementation
        that can solve the equation of motion.
    times : Array
        One-dimensional vector of timestamps.
    """

    __propagation: Propagation

    def __init__(self, propagation: Propagation, times: Array):
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

    def measure_normalised_scalar(self) -> float:
        """Return the result of measurement.

        Returns
        -------
        Array
            The result of the measurement.

        """
        return float(np.random.random())
