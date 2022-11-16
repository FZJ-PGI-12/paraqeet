import numpy as np
from typing import List
from scipy.optimize import minimize

from cthree.Optimiser import Optimiser
from cthree.Optimisable import Optimisable
from cthree.measurement.Measurement import Measurement


class ScipyOptimiser(Optimiser):
    _measure: Measurement
    __optimisables: List[Optimisable]
    __optim_vector: np.ndarray

    def __init__(self, measure: Measurement, optimisables: List[Optimisable]):
        self._measure = measure
        self.__optimisables = optimisables

    def setup_optim(self):
        # What parameters do I want?
        self.__optim_vector = ...

    def optimise(self):
        return minimize(
            fun=self.set_parameters,
            x0=self.__optimisables.getParameters()
        )

    def set_parameters(self, values) -> None:
        self.optim_vector = values
        self._measure.measure()
