from typing import List, Optional

from cthree.Optimisable import Optimisable
from cthree.measurement.Measurement import Measurement


class Optimiser:
    _measure: Measurement
    _optimisables: List[Optimisable]

    def __init__(self, measure: Measurement, optimisables: Optional[List[Optimisable]] = None):
        self._measure = measure
        self._optimisables = optimisables or measure.getParameters()

    def optimise(self):
        pass
