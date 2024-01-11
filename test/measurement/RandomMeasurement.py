from typing import List

import numpy as np

from cthree.Quantity import Quantity
from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation


class RandomMeasurement(Measurement):
    """
    Mock class that returns a random measurement value between 0 and 1.
    """

    __propagation: Propagation

    def __init__(self, propagation: Propagation, times: np.ndarray):
        super().__init__(times=times)
        self.__propagation = propagation

    def getParameters(self) -> List[Quantity]:
        return []

    def measure(self) -> np.ndarray:
        return np.random.random()
