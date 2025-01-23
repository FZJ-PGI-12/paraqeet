"""Class definition for a mock system that always returns the same value."""

import numpy as np

from cthree.Quantity import Quantity
from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation


class ConstantMeasurement(Measurement):
    """Mock implementation that always returns the same value.

    Parameters
    ----------
    propagation : cthree.propagation.Propagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    value : float, default=1.0
        Value of the measurement.
    times : float | None, optional
        Time variable value.
    """

    __propagation: Propagation
    __value: float

    def __init__(
        self,
        propagation: Propagation,
        value: float = 1.0,
        times: float | None = None,
    ):
        super().__init__(times=times)
        self.__propagation = propagation
        self.__value = value

    def get_parameters(self) -> list[Quantity]:
        """Get the system parameters.

        Parameters
        ----------
        list[cthree.Quantity]
            List of parameters of the system.

        """
        return []

    def measure(self) -> np.ndarray:
        """Get the measurement value.

        Returns
        -------
        numpy.ndarray
            The value of the measurement.

        """
        return self.__value
