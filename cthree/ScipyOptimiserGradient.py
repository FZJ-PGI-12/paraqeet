from typing import Dict, List

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from cthree.Optimiser import Optimiser
from cthree.Quantity import Quantity
from cthree.measurement.Measurement import Measurement


class ScipyOptimiserGradient(Optimiser):
    """
    Minimize the outcome of a measuremnt with the scipy optimisation package.
    """

    _measure: Measurement
    _optimisables: List[Quantity]
    __opt_idxs: List[int]
    __options: Dict
    __method: str

    def __init__(
        self, measure: Measurement, optimisables: List[Quantity] | None = None
    ):
        super().__init__(measure, optimisables)
        self.__options = {"disp": True}
        self.__method = "L-BFGS-B"

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

    def setMethod(self, method: str):
        self.__method = method

    def setOptions(self, opts: Dict):
        self.__options = opts

    def updateOption(self, key, val):
        self.__options.update(key, val)

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
        return 1 - self._measure.measureNormalised()

    def _setParametersAndMeasureJac(self, values) -> float:
        """
        Update the parameter values and return the gradient of a measurement result. Internal callback.
        """
        for index, val in enumerate(np.split(values, self.__opt_idxs[:-1])):
            self._optimisables[index].setReducedValue(val)
        return -1 * self._measure.measureGradient()
