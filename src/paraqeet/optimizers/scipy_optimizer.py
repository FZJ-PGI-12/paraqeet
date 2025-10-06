"""Class definition of the Scipy optimizer model."""

from collections.abc import Callable


from scipy.optimize import minimize
import numpy as np
from paraqeet.measurement.measurement import Measurement
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.optimizer import OptimizationResult, Optimizer


class ScipyOptimizer(Optimizer):
    """Minimize the outcome of a measuremnt with the scipy optimization package.

    Parameters
    ----------
    measure: Measurement
        Implementation of the Measurement class that measures the observable
        to be minimised.
    optimizable: OptimizationMap
        An optimization map containing all parameters that can be optimized.

    """

    _measure: Measurement
    _opt_idxs: list[int]
    _options: dict
    _method: str
    _callback: Callable | None

    def __init__(self, measure: Measurement, optimizables: OptimizationMap):
        super().__init__(measure, optimizables)
        self._options = {"disp": True}
        self._method = "L-BFGS-B"
        self._callback = None

    @property
    def method(self) -> str:
        """Returns the currently selected optimization method."""
        return self._method

    @method.setter
    def method(self, method: str) -> None:
        """Select method from scipy.optimize.minimize.

        See Also
        --------
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        Parameters
        ----------
        method: str
            Type of solver, specified by string value.

        """
        self._method = method

    def set_options(self, opts: dict):
        """Set the options for the system."""
        self._options.update(opts)

    def update_option(self, key, val):
        """Updates one option for the system."""
        self._options[key] = val

    @property
    def callback(self) -> Callable | None:
        """Returns the callback function."""
        return self._callback

    @callback.setter
    def callback(self, cbfun: Callable) -> None:
        """Set the callback function for the optimizer.

        Parameters
        ----------
        Callable
            The function to be set as the callback.

        """
        self._callback = cbfun

    def optimize(self) -> OptimizationResult:
        """Optimize the system via the Scipy optimizer.

        Performs the actual optimization.

        Since the search parameters are dimensionless and bound by [-1, 1], we set the bounds of the scipy minimize
        module to -1, and 1 explicitely in each search dimension.

        Returns
        -------
        OptimizationResult
            The result of the optimization.

        """
        if self._logger:
            self._logger.start()

        self._build_optimizable_index_list()
        self._optimizables.register_params_with_optimizables()

        # Collect the initial values of all parameters
        init = []
        for qty in self._optimizables.get_all_parameters():
            init.append(qty.get_reduced_value())  # reduced values are between [-1, 1]

        opt_res = minimize(
            fun=self._set_parameters_and_measure,
            x0=np.concatenate(init).flatten(),
            bounds=[(-1, 1)] * self._opt_idxs[-1],
            method=self._method,
            options=self._options,
            callback=self._callback,
        )

        if self._logger:
            self._logger.stop(str(opt_res))

        return OptimizationResult(
            status=OptimizationResult.STATUS_SUCCESS if opt_res.success else OptimizationResult.STATUS_FAILED,
            value=opt_res.fun,
            iterations=opt_res.nfev,
            message=opt_res.message,
            raw_result=opt_res,
        )

    def _set_parameters_and_measure(self, values) -> float:
        """Update the parameter values and return the measurement result.

        Internal callback.

        Parameters
        ----------
        values: Array
            Parameter values for the update.

        Returns
        -------
        Array
            Returns the measurement result.

        """
        log = []
        params = self._optimizables.get_all_parameters()
        for index, val in enumerate(np.split(values, self._opt_idxs[:-1])):
            params[index].set_reduced_value(val)
            log.append(params[index])
        infid = 1 - self._measure.measure_normalised_scalar()

        if self._logger:
            self._logger.log(log, infid)
        return infid
