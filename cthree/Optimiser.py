from typing import List

from cthree.Optimisable import Optimisable
from cthree.measurement.Measurement import Measurement


class Optimiser:
    _measure: Measurement
    __optimisables: List[Optimisable]

    def __init__(self, measure: Measurement, optimisables: List[Optimisable]):
        self._measure = measure
        self.__optimisables = optimisables

    def optimise(self):
        # crazy stuff happens here
        pass
