from typing import Dict, List

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from cthree.ScipyOptimiser import ScipyOptimiser
from cthree.measurement.Measurement import Measurement


class ScipyOptimiserGradient(ScipyOptimiser):
    """
    Minimize the outcome of a measuremnt with the scipy optimisation package.
    """

    _measure: Measurement
    _opt_idxs: List[int]
    _options: Dict
    _method: str

    def optimise(self) -> OptimizeResult:
        if self._logger:
            self._logger.start()

        self._buildOptimisableIndexList()
        self._optimisables.registerParamsWithOptimisables()

        init = []
        for qty in self._optimisables.getAllParameters():
            init.append(qty.getReducedValue())
        result = minimize(
            fun=self._setParametersAndMeasure,
            jac=self._setParametersAndMeasureJac,
            x0=np.concatenate(init).flatten(),
            bounds=[(-1, 1)] * self._opt_idxs[-1],
            method=self._method,
            options=self._options,
        )

        if self._logger:
            self._logger.stop(str(result))
        return result

    def _setParametersAndMeasureJac(self, values) -> np.ndarray:
        """
        Update the parameter values and return the gradient of a measurement result. Internal callback.
        """
        params = self._optimisables.getAllParameters()
        for index, val in enumerate(np.split(values, self._opt_idxs[:-1])):
            params[index].setReducedValue(val[0])
        return -1 * self._measure.measureGradient()
