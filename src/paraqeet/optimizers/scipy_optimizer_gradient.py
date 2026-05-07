"""Class definition for the Scipy optimizer gradient model."""

from typing import Callable

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from paraqeet.exceptions import IncompatibleOptimizationMap
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.optimizer import OptimizationResult
from paraqeet.optimizers.scipy_optimizer import ScipyOptimizer
from paraqeet.quantity import Array, Float


class ScipyOptimizerGradient(ScipyOptimizer):
    """The Scipy Optimizer gradient model.

    Minimize the outcome of a measurement with the Scipy optimization package by providing gradient values.

    Parameters
    ----------
    measure_and_gradient_func: Callable[[Array], tuple[Float, Array]]
        Function implementing measurement of observables to be minimized.
    optimization_map: OptimizationMap
        An optimization map containing all parameters that can be optimized.

    """

    _grad_cache: Array  # of shape (n_parameters,)
    _scales: Array
    _measure_and_gradient_func: Callable[[Array], tuple[Float, Array]]

    def __init__(
        self, measure_and_gradient_func: Callable[[Array], tuple[Float, Array]], optimization_map: OptimizationMap
    ) -> None:
        self._measure_and_gradient_func = measure_and_gradient_func
        self._measure_func = lambda times: self._measure_and_gradient_func(times)[0]
        super().__init__(self._measure_func, optimization_map)
        params = self._optimization_map.get_all_parameters()
        self._scales = jnp.array([p.get_scale() for p in params]).flatten()

    def optimize(self, times: Array | float) -> OptimizationResult:
        """Optimize via the Scipy optimizer gradient model.

        Performs the actual optimization.

        *Note - If input `times` is a float, then the start time of propagation is implicity assumed to be zero.
        For an array of times, the first time point is the start time.*

        Returns
        -------
        OptimizationResult
            The result of the optimization.

        """
        self._times = jnp.array([0.0, times]) if isinstance(times, float) else times

        if self._logger:
            self._logger.start()

        self._build_optimizable_index_list()
        self._optimization_map.register_params_with_optimizables()

        init = []
        for qty in self._optimization_map.get_all_parameters():
            init.append(qty.get_reduced_value())

        try:
            result = minimize(
                fun=self._set_parameters_and_measure,
                jac=self._lookup_jac,
                x0=jnp.concatenate(init).flatten(),
                bounds=[(-1, 1)] * self._opt_idxs[-1],
                method=self._method,
                options=self._options,
                callback=self._callback,
            )
        except Exception as e:
            if "_lbfgsb._lbfgsb.setulb: failed to create array from the 7th" + " argument `g`" in str(e):
                raise IncompatibleOptimizationMap(
                    "Number of quantities in optMap differ from number of" + f" gradients computed. \n {e}"
                )
            else:
                raise e

        if self._logger:
            self._logger.stop(str(result))

        return OptimizationResult(
            status=(OptimizationResult.STATUS_SUCCESS if result.success else OptimizationResult.STATUS_FAILED),
            value=result.fun,
            iterations=result.nfev,
            message=result.message,
            raw_result=result,
        )

    def _set_parameters_and_measure(self, values) -> float:
        """Update the parameter values and return measurement result.

        Returns the measurement result including gradient.
        The gradient is stored in a local cache for lookup.
        This tailored for L-BFGS-B or similar algorithms that alternate
        between function and gradient calls.
        Internal callback.

        Parameters
        ----------
        values: Array
            Parameter values for the update.

        Returns
        -------
        Array
            Returns the inverse of the fidelity.

        """
        log = []
        params = self._optimization_map.get_all_parameters()
        for index, val in enumerate(np.split(values, self._opt_idxs[:-1])):  # TODO: Convert to jax
            params[index].set_reduced_value(val)
            log.append(params[index])

        fun, grad = self._measure_and_gradient_func(self._times)
        self._grad_cache = grad

        infid = 1.0 - fun
        if self._logger:
            self._logger.log(log, float(infid))
        return float(1 - fun)

    def _lookup_jac(self, values) -> Array:
        """Update the parameter values.

        Return the gradient of a measurement result.
        Internal callback.

        Parameters
        ----------
        values: Array
            Parameter values for the update.

        Returns
        -------
        Array
            Returns the gradient of a measurement result.

        """
        return -1 * self._grad_cache * self._scales
