from typing import Dict, List

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from cthree.ScipyOptimiser import ScipyOptimiser
from cthree.measurement.Measurement import Measurement


class ScipyOptimiserGradient(ScipyOptimiser):
    """
    Minimize the outcome of a measurement with the scipy optimisation package.
    """

    _measure: Measurement
    _opt_idxs: List[int]
    _options: Dict
    _method: str
    __gradCache: np.ndarray  # of shape (n_parameters,)

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
            jac=self._lookupJac,
            x0=np.concatenate(init).flatten(),
            bounds=[(-1, 1)] * self._opt_idxs[-1],
            method=self._method,
            options=self._options,
        )

        if self._logger:
            self._logger.stop(str(result))
        return result

    def _setParametersAndMeasure(self, values) -> float:
        """
        Update the parameter values and return the measurement result including gradient.
        The gradient is stored in a local cache for lookup. This tailored for L-BFGS-B or
        similar algorithms that alternate between function and gradient calls.
        Internal callback.
        """
        log = []
        params = self._optimisables.getAllParameters()
        for index, val in enumerate(np.split(values, self._opt_idxs[:-1])):
            params[index].setReducedValue(val)
            log.append(params[index])
        fun, grad = self._measure.measureWithGradient()
        self.__gradCache = grad

        infid = 1 - fun
        if self._logger:
            self._logger.log(log, infid)
        return 1 - fun

    def _lookupJac(self, values) -> np.ndarray:
        """
        Update the parameter values and return the gradient of a measurement result. Internal callback.
        """
        return -1 * self.__gradCache
