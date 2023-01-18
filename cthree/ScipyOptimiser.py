from typing import List
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
            init.append(qty.getValue())
        return minimize(
            fun=self.setParameters,
            x0=init
        )

    def setParameters(self, values) -> float:
        for index, val in enumerate(values):
            self._optimisables[index].setValue(val)
        return self._measure.measure()
