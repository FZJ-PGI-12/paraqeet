from typing import List

import numpy as np

from cthree.Quantity import Quantity
from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation


class ConstantMeasurement(Measurement):
    """
    Mock implementation that always returns the same value.
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

    def getParameters(self) -> List[Quantity]:
        return []

    def measure(self) -> np.ndarray:
        return self.__value
