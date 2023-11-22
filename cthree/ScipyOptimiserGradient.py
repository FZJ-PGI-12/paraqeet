from typing import List

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from cthree.Optimiser import Optimiser
from cthree.Quantity import Quantity
from cthree.measurement.Measurement import Measurement


class ScipyOptimiser(Optimiser):
    """
    Minimize the outcome of a measuremnt with the scipy optimisation package.
    """

    _measure: Measurement
    _optimisables: List[Quantity]
    __opt_idxs: List[int]

    def optimise(self) -> OptimizeResult:
        init = []
        for qty in self._optimisables:
            init.append(qty.getReducedValue())
        return minimize(
            fun=self._setParametersAndMeasure,
            jac=self._setParametersAndMeasureJac,
            x0=np.concatenate(init).flatten(),
            bounds=[(-1, 1)] * self.__opt_idxs[-1],
            method="L-BFGS-B",
            options={"disp": True},
        )

    def setOptimisables(self, opt: List[Quantity]) -> None:
        """
        Registers optimisables and their length to keep track of vector and matrix valued parameters.
        """
        super().setOptimisables(opt)
        self.__opt_idxs = []
        index = 0
        for qty in opt:
            index += qty.getLength()
            self.__opt_idxs.append(index)

    def _setParametersAndMeasure(self, values) -> float:
        """
        Update the parameter values and return the measurement result. Internal callback.
        """
        for index, val in enumerate(np.split(values, self.__opt_idxs[:-1])):
            self._optimisables[index].setReducedValue(val)
        return 1 - self._measure.measure()

    def _setParametersAndMeasureJac(self, values) -> float:
        """
        Update the parameter values and return the measurement result. Internal callback.
        """
        for index, val in enumerate(np.split(values, self.__opt_idxs[:-1])):
            self._optimisables[index].setReducedValue(val)
        return self._measure.measureGradient()
