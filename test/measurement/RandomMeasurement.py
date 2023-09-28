from typing import Tuple, List

import numpy as np

from cthree.Quantity import Quantity
from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation
from cthree.Exceptions import IncompatibleLayersException


class RandomMeasurement(Measurement):
    """
    Mock class that returns a random measurement value between 0 and 1.
    """
    __propagation: Propagation

    def __init__(self, propagation: Propagation):
        super().__init__()
        self.__propagation = propagation

    def getParameters(self) -> List[Quantity]:
        return []

    def measure(self) -> float:
        return np.random.random()
