"""Class definition of the Scipy optimiser model."""

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from cthree.optimisation_map import OptimisationMap
from cthree.optimisers.optimiser import Optimiser, OptimisationResult
from cthree.measurement.measurement import Measurement


class ScipyOptimiser(Optimiser):
    """Minimize the outcome of a measuremnt with the scipy optimisation package.

    Parameters
    ----------
    measure : cthree.measurement.measurement
        Implementation of the Measurement class that measures the observable
        to be minimised.
    optimisable : cthree.optimisation_map
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

        Returns
        -------
        cthree.optimisers.optimiser.OptimisationResult
            The result of the optimisation.

        """
        if self._logger:
            self._logger.start()

        self._build_optimisable_index_list()
        self._optimisables.register_params_with_optimisables()

        # Collect the initial values of all parameters
        init = []
        for qty in self._optimisables.get_all_parameters():
            init.append(qty.get_reduced_value())

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

        return OptimisationResult(
            status=OptimisationResult.STATUS_SUCCESS if opt_res.success else OptimisationResult.STATUS_FAILED,
            value=opt_res.fun,
            iterations=opt_res.nfev,
            message=opt_res.message,
            raw_result=opt_res,
        )

    def _set_parameters_and_measure(self, values) -> np.ndarray:
        """Update the parameter values and return the measurement result.

        Internal callback.

        Parameters
        ----------
        values : numpy.ndarray
            Parameter values for the update.

        Returns
        -------
        numpy.ndarray
            Returns the measurement result.

        """
        log = []
        params = self._optimisables.get_all_parameters()
        for index, val in enumerate(np.split(values, self._opt_idxs[:-1])):
            params[index].set_reduced_value(val)
            log.append(params[index])
        infid = 1 - self._measure.measure_normalised()

        if self._logger:
            self._logger.log(log, float(infid))
        return infid
