from typing import Callable, Dict, List

import numpy as np
from scipy.optimize import minimize

from cthree.OptimisationMap import OptimisationMap
from cthree.optimisers.Optimiser import Optimiser, OptimisationResult
from cthree.measurement.Measurement import Measurement


class ScipyOptimiser(Optimiser):
    """
    Minimize the outcome of a measuremnt with the scipy optimisation package.
    """

    _measure: Measurement
    _opt_idxs: List[int]
    _options: Dict
    _method: str
    _callback: Callable | None

    def __init__(self, measure: Measurement, optimisables: OptimisationMap):
        super().__init__(measure, optimisables)
        self._options = {"disp": True}
        self._method = "L-BFGS-B"
        self._callback = None

    def setMethod(self, method: str):
        """Select method from scipy.optimize.minimize.
        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        Parameters
        ----------
        method : str
        """
        self._method = method

    def setOptions(self, opts: Dict):
        self._options.update(opts)

    def updateOption(self, key, val):
        self._options.update(key, val)

    def setCallback(self, cbfun: Callable) -> None:
        self._callback = cbfun

    def optimise(self) -> OptimisationResult:
        if self._logger:
            self._logger.start()

        self._buildOptimisableIndexList()
        self._optimisables.registerParamsWithOptimisables()

        # Collect the initial values of all parameters
        init = []
        for qty in self._optimisables.getAllParameters():
            init.append(qty.getReducedValue())

        opt_res = minimize(
            fun=self._setParametersAndMeasure,
            x0=np.concatenate(init).flatten(),
            bounds=[(-1, 1)] * self._opt_idxs[-1],
            method=self._method,
            options=self._options,
            callback=self._callback,
        )

        if self._logger:
            self._logger.stop(str(opt_res))

        self._rawResult = opt_res
        return OptimisationResult(
            status=OptimisationResult.STATUS_SUCCESS if opt_res.success else OptimisationResult.STATUS_FAILED,
            value=opt_res.fun,
            iterations=opt_res.nfev,
            message=opt_res.message,
        )

    def _setParametersAndMeasure(self, values) -> np.ndarray:
        """
        Update the parameter values and return the measurement result. Internal callback.
        """
        log = []
        params = self._optimisables.getAllParameters()
        for index, val in enumerate(np.split(values, self._opt_idxs[:-1])):
            params[index].setReducedValue(val)
            log.append(params[index])
        infid = 1 - self._measure.measureNormalised()

        if self._logger:
            self._logger.log(log, infid)
        return infid
