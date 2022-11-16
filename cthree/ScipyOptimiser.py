from typing import List
from scipy.optimize import minimize

from cthree.Optimiser import Optimiser
from cthree.Quantity import Quantity
from cthree.measurement.Measurement import Measurement


class ScipyOptimiser(Optimiser):
    _measure: Measurement
    __optimisables: List[Quantity]

    def __init__(self, measure: Measurement, optimisables: List[Quantity]):
        self._measure = measure
        self.__optimisables = optimisables

    def optimise(self):
        init = []
        for qty in self.__optimisables:
            init.append(qty.get_value())
        return minimize(
            fun=self.set_parameters,
            x0=init
        )

    def set_parameters(self, values) -> float:
        for index, val in enumerate(values):
            self.__optimisables[index].set_value(val)
        return self._measure.measure()
