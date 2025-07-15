"""Class definition for the Scipy optimiser gradient model."""

import numpy as np
import jax.numpy as jnp
from paraqeet.quantity import Array
from scipy.optimize import minimize

from paraqeet.exceptions import IncompatibleOptimisationMap
from paraqeet.measurement.measurement import Measurement
from paraqeet.optimisation_map import OptimisationMap
from paraqeet.optimisers.optimiser import OptimisationResult
from paraqeet.optimisers.scipy_optimiser import ScipyOptimiser


class ScipyOptimiserGradient(ScipyOptimiser):
    """The Scipy Optimiser gradient model.

    Minimize the outcome of a measurement with the
    Scipy optimisation package.

    """

    __gradCache: Array  # of shape (n_parameters,)
    __scales: Array

    def __init__(self, measure: Measurement, optimisables: OptimisationMap) -> None:
        super().__init__(measure, optimisables)
        params = self._optimisables.get_all_parameters()
        self.__scales = jnp.array([p.get_scale() for p in params]).flatten()

    def optimise(self) -> OptimisationResult:
        """Optimise via the Scipy optimizer gradient model.

        Performs the actual optimisation.

        Returns
        -------
        paraqeet.optimisers.optimiser.OptimisationResult
            The result of the optimisation.

        """
        if self._logger:
            self._logger.start()

        self._build_optimisable_index_list()
        self._optimisables.register_params_with_optimisables()

        init = []
        for qty in self._optimisables.get_all_parameters():
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
                raise IncompatibleOptimisationMap(
                    "Number of quantities in optMap differ from number of" + f" gradients computed. \n {e}"
                )
            else:
                raise e

        if self._logger:
            self._logger.stop(str(result))

        return OptimisationResult(
            status=(OptimisationResult.STATUS_SUCCESS if result.success else OptimisationResult.STATUS_FAILED),
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
        values : Array
            Parameter values for the update.

        Returns
        -------
        Array
            Returns the inverse of the fidelity.

        """
        log = []
        params = self._optimisables.get_all_parameters()
        for index, val in enumerate(np.split(values, self._opt_idxs[:-1])):  # TODO: Convert to jax
            params[index].set_reduced_value(val)
            log.append(params[index])
        fun, grad = self._measure.measure_with_gradient()
        self.__gradCache = grad

        infid = 1 - fun
        if self._logger:
            self._logger.log(log, infid)
        return 1 - fun

    def _lookup_jac(self, values) -> Array:
        """Update the parameter values.

        Return the gradient of a measurement result.
        Internal callback.

        Parameters
        ----------
        values : Array
            Parameter values for the update.

        Returns
        -------
        Array
            Returns the gradient of a measurement result.

        """
        return -1 * self.__gradCache * self.__scales
