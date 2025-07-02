"""Class definition of the Scipy optimiser model."""

from collections.abc import Callable


from scipy.optimize import minimize
import numpy as np
from paraqeet.measurement.measurement import Measurement
from paraqeet.optimisation_map import OptimisationMap
from paraqeet.optimisers.optimiser import OptimisationResult, Optimiser


class ScipyOptimiser(Optimiser):
    """Minimize the outcome of a measuremnt with the scipy optimisation package.

    Parameters
    ----------
    measure : Measurement
        Implementation of the Measurement class that measures the observable
        to be minimised.
    optimisable : OptimisationMap
        An optimisation map containing all parameters that can be optimised.

    """

    _measure: Measurement
    _opt_idxs: list[int]
    _options: dict
    _method: str
    _callback: Callable | None

    def __init__(self, measure: Measurement, optimisables: OptimisationMap):
        super().__init__(measure, optimisables)
        self._options = {"disp": True}
        self._method = "L-BFGS-B"
        self._callback = None

    @property
    def method(self) -> str:
        """Returns the currently selected optimisation method."""
        return self._method

    @method.setter
    def method(self, method: str) -> None:
        """Select method from scipy.optimize.minimize.

        See Also
        --------
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        Parameters
        ----------
        method : str
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
        """Set the callback function for the optimiser.

        Parameters
        ----------
        collections.abc.Callable
            The function to be set as the callback.

        """
        self._callback = cbfun

    def optimise(self) -> OptimisationResult:
        """Optimise the system via the Scipy optimizer.

        Performs the actual optimisation.

        Since the search parameters are dimensionless and bound by [-1, 1], we set the bounds of the scipy minimize
        module to -1, and 1 explicitely in each search dimension.

        Returns
        -------
        paraqeet.optimisers.optimiser.OptimisationResult
            The result of the optimisation.

        """
        if self._logger:
            self._logger.start()

        self._build_optimisable_index_list()
        self._optimisables.register_params_with_optimisables()

        # Collect the initial values of all parameters
        init = []
        for qty in self._optimisables.get_all_parameters():
            init.append(qty.get_reduced_value())  # reduced values are between [-1, 1]

        opt_res = minimize(
            fun=self._set_parameters_and_measure,
            x0=np.concatenate(init).flatten(),
            bounds=[(-1, 1)] * len(self._opt_idxs),  # len(.) gives the number of parameters
            method=self._method,
            options=self._options,
            callback=self._callback,
        )

        if self._logger:
            self._logger.stop(str(opt_res))

        return OptimisationResult(
            status=OptimisationResult.STATUS_SUCCESS if opt_res.success else OptimisationResult.STATUS_FAILED,
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
        values : Array
            Parameter values for the update.

        Returns
        -------
        Array
            Returns the measurement result.

        """
        log = []
        params = self._optimisables.get_all_parameters()
        for index, val in enumerate(np.split(values, self._opt_idxs[:-1])):
            params[index].set_reduced_value(val)
            log.append(params[index])
        infid = 1 - self._measure.measure_normalised_scalar()

        if self._logger:
            self._logger.log(log, infid)
        return infid
