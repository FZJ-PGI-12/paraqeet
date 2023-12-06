from typing import Dict, List

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
    _opt_idxs: List[int]
    _options: Dict
    _method: str

    def __init__(
        self, measure: Measurement, optimisables: List[Quantity] | None = None
    ):
        super().__init__(measure, optimisables)
        self._options = {"disp": True}
        self._method = "L-BFGS-B"

    def setMethod(self, method: str):
        """Select method from scipy.optimize.minimize.
        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        Parameters
        ----------
        method : str
        """
        self._method = method

    def setOptions(self, opts: Dict):
        self._options = opts

    def updateOption(self, key, val):
        self._options.update(key, val)

    def optimise(self) -> OptimizeResult:
        if self._logger:
            self._logger.start()

        init = []
        for qty in self._optimisables:
            init.append(qty.getReducedValue())
        opt_res = minimize(
            fun=self._setParametersAndMeasure,
            x0=np.concatenate(init).flatten(),
            bounds=[(-1, 1)] * self._opt_idxs[-1],
            method=self._method,
            options=self._options,
        )

        if self._logger:
            self._logger.stop(str(opt_res))

        return opt_res

    def setOptimisables(self, opt: List[Quantity]) -> None:
        """
        Registers optimisables and their length to keep track of vector and matrix valued parameters.
        """
        super().setOptimisables(opt)
        self._opt_idxs = []
        index = 0
        for qty in opt:
            index += qty.getLength()
            self._opt_idxs.append(index)

    def _setParametersAndMeasure(self, values) -> float:
        """
        Update the parameter values and return the measurement result. Internal callback.
        """
        log = []
        for index, val in enumerate(np.split(values, self.__opt_idxs[:-1])):
            self._optimisables[index].setReducedValue(val)
            log.append(self._optimisables[index])
        infid = 1 - self._measure.measureNormalised()

        if self._logger:
            self._logger.log(log, infid)
        return infid
