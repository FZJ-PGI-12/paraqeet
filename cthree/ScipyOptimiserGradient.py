from typing import Dict, List

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from cthree.ScipyOptimiser import ScipyOptimiser
from cthree.Quantity import Quantity
from cthree.measurement.Measurement import Measurement


class ScipyOptimiserGradient(ScipyOptimiser):
    """
    Minimize the outcome of a measuremnt with the scipy optimisation package.
    """

    _measure: Measurement
    _optimisables: List[Quantity]
    __opt_idxs: List[int]
    __options: Dict
    __method: str

    def optimise(self) -> OptimizeResult:
        init = []
        for qty in self._optimisables:
            init.append(qty.getReducedValue())
        return minimize(
            fun=self._setParametersAndMeasure,
            jac=self._setParametersAndMeasureJac,
            x0=np.concatenate(init).flatten(),
            bounds=[(-1, 1)] * self.__opt_idxs[-1],
            method=self.__method,
            options=self.__options,
        )

    def _setParametersAndMeasureJac(self, values) -> float:
        """
        Update the parameter values and return the gradient of a measurement result. Internal callback.
        """
        for index, val in enumerate(np.split(values, self.__opt_idxs[:-1])):
            self._optimisables[index].setReducedValue(val)
        return -1 * self._measure.measureGradient()
