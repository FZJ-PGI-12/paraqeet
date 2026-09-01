"""Class definition of the Scipy optimizer model."""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.optimize import OptimizeResult, minimize

from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.optimizer import OptimizationResult, Optimizer
from paraqeet.quantity import Array, Float


class ScipyOptimizer(Optimizer):
    """Minimize the outcome of a measurement with the scipy optimization package."""

    _measure_func: Callable[[Array], Float]
    _opt_idxs: list[int]
    _options: dict
    _method: str
    _callback: Callable | None
    _num_iterations: int = 0

    def __init__(self, measure_func: Callable[[Array], Float], optimization_map: OptimizationMap) -> None:
        """
        Args:
            measure_func: Function implementing measurement of observables
                to be minimized.
            optimization_map: An optimization map containing all parameters
                that can be optimized.
        """
        super().__init__(measure_func, optimization_map)
        self._options = {}
        self._method = "L-BFGS-B"
        self._callback = self._default_callback

    @property
    def method(self) -> str:
        """Returns the currently selected optimization method."""
        return self._method

    @method.setter
    def method(self, method: str) -> None:
        """Select method from scipy.optimize.minimize.

        Args:
            method: Type of solver, specified by string value.
        """
        self._method = method

    def set_options(self, opts: dict) -> None:
        """Set the options for the system."""
        self._options.update(opts)

    def update_option(self, key: str, val: Any) -> None:
        """Update one option for the system."""
        self._options[key] = val

    @property
    def callback(self) -> Callable | None:
        """Returns the callback function."""
        return self._callback

    @callback.setter
    def callback(self, cbfun: Callable) -> None:
        """Set the callback function for the optimizer.

        Args:
            cbfun: The function to be set as the callback.
        """
        self._callback = cbfun

    def _default_callback(self, intermediate_result: OptimizeResult) -> None:
        self._num_iterations += 1
        fun = intermediate_result.fun if hasattr(intermediate_result, "fun") else None
        if self._num_iterations % 10 == 0:
            print(f"Iteration {self._num_iterations:4d} | Infid = {fun:.6e}")

    def optimize(self, times: Array | float) -> OptimizationResult:
        """Optimize the system via the Scipy optimizer.

        Performs the actual optimization.

        Since the search parameters are dimensionless and bound by [-1, 1],
        we set the bounds of the scipy minimize module to -1 and 1 explicitly
        in each search dimension.

        Note:
            If input ``times`` is a float, then the start time of propagation
            is implicitly assumed to be zero. For an array of times, the first
            time point is the start time.

        Args:
            times: Array of times or a float (assumed start time zero).

        Returns:
            The result of the optimization.
        """
        if self._logger:
            self._logger.start()

        self._times = jnp.array([0.0, times]) if isinstance(times, float) else times

        self._build_optimizable_index_list()
        self._optimization_map.register_params_with_optimizables()

        # Collect the initial values of all parameters
        init = []
        for qty in self._optimization_map.get_all_parameters():
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
            iterations=opt_res.nit,
            evaluations=opt_res.nfev,
            message=opt_res.message,
            raw_result=opt_res,
        )

    def _set_parameters_and_measure(self, values: Array) -> Float:
        """Update the parameter values and return the measurement result.

        Internal callback.

        Args:
            values: Parameter values for the update.

        Returns:
            The infidelity, i.e. one minus the measurement result (the optimizer minimizes this).
        """
        log = []
        params = self._optimization_map.get_all_parameters()
        for index, val in enumerate(np.split(values, self._opt_idxs[:-1])):
            params[index].set_reduced_value(val)
            log.append(params[index])
        infid = 1 - self._measure_func(self._times)

        if self._logger:
            self._logger.log(log, float(infid))
        return infid
