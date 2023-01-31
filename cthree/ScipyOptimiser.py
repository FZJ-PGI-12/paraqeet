from typing import List

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from cthree.Optimiser import Optimiser
from cthree.Quantity import Quantity
from cthree.measurement.Measurement import Measurement


class ScipyOptimiser(Optimiser):
    _measure: Measurement
    _optimisables: List[Quantity]

    def optimise(self) -> OptimizeResult:
        init = []
        for qty in self._optimisables:
            init.append(qty.getReducedValue())
        return minimize(
            fun=self.setParameters,
            x0=np.array(init),
            bounds=[(-1, 1)] * len(self._optimisables),
        )

    def setParameters(self, values) -> float:
        for index, val in enumerate(values):
            self._optimisables[index].setReducedValue(val)
        return self._measure.measure()
