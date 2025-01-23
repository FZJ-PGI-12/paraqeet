"""Class definition of the CMA-Es Optimiser model."""

from collections.abc import Callable
from cthree.FileLogger import Logger
from cthree.OptimisationMap import OptimisationMap
from cthree.measurement.Measurement import Measurement
from cthree.optimisers.Optimiser import Optimiser, OptimisationResult
import numpy as np
import cma.evolution_strategy as cma


class CMAEsOptimiser(Optimiser):
    """Wrapper for the pycma implementation of CMA-Es.

    The pycmi implementation has the following custom options for optimisation:
    noise : float
        Artificial noise added to a function evaluation.
    init_point : boolean
        Force the use of the initial point in the first generation.
    spread : float
        Adjust the parameter spread of the first generation cloud.
    stop_at_convergence : int
        Custom stopping condition. Stop if the cloud shrunk for this number of
        generations.
    stop_at_sigma : float
        Custom stopping condition. Stop if the cloud shrunk to this standard
        deviation.

    See also: http://cma.gforge.inria.fr/apidocs-pycma/

    Parameters
    ----------
    measure : cthree.measurement.Measurement
        Represents any observable and the process of measurement itself.
    optimisables : cthree.OptimisationMap
        Optimisable interface for all parameters considered in optimisation.
    logger : cthree.FileLogger | None, default=None
        The file logger object.
    callback
        Callback function for optimisation.

    """

    _options: dict
    callback: Callable | None

    def __init__(
        self,
        measure: Measurement,
        optimisables: OptimisationMap,
        logger: Logger | None = None,
        callback=None,
    ):
        super().__init__(measure, optimisables, logger)
        self._options = {
            "noise": 0,
            "batch_noise": 0,
            "init_point": False,
            "spread": 0.1,
            "bounds": [-1.0, 1],
        }
        self.setCallback(callback)

    def getOptions(self) -> dict:
        """Get options from the system."""
        return self._options

    def setOptions(self, opts):
        """Set options for the system."""
        self._options.update(opts)

    def setCallback(self, cbfun: Callable) -> None:
        """Set the callback function for the optimiser.

        Parameters
        ----------
        collections.abc.Callable
            The function to be set as the callback.

        """
        self._callback = cbfun

    def optimise(self) -> OptimisationResult:
        """Optimise the system via the CMA-Es optimiser.

        Performs the actual optimisation via the following custom options:
        noise : float
            Artificial noise added to a function evaluation.
        init_point : boolean
            Force the use of the initial point in the first generation.
        spread : float
            Adjust the parameter spread of the first generation cloud.
        stop_at_convergence : int
            Custom stopping condition. Stop if the cloud shrunk for this number
            of generations.
        stop_at_sigma : float
            Custom stopping condition. Stop if the cloud shrunk to this
            standard deviation.

        Returns
        -------
        cthree.optimisers.Optimiser.OptimisationResult
            Result of optimization via the OptimisationResult object.
            (status, value, iterations and the raw result)

        """
        options = {}
        options.update(self._options)
        options = self._options
        if "noise" in options:
            noise = float(options.pop("noise"))

        if "batch_noise" in options:
            batch_noise = float(options.pop("batch_noise"))

        if "init_point" in options:
            init_point = bool(options.pop("init_point"))

        if "spread" in options:
            spread = float(options.pop("spread"))

        shrunk_check = False
        if "stop_at_convergence" in options:
            sigma_conv = int(options.pop("stop_at_convergence"))
            sigmas = []
            shrunk_check = True

        sigma_check = False
        if "stop_at_sigma" in options:
            stop_sigma = int(options.pop("stop_at_sigma"))
            sigma_check = True

        settings = options

        if self._logger:
            self._logger.start()

        self._buildOptimisableIndexList()
        self._optimisables.register_params_with_optimisables()

        x_init = []
        for qty in self._optimisables.get_all_parameters():
            x_init.append(qty.get_reduced_value())

        es = cma.CMAEvolutionStrategy(np.concatenate(x_init).flatten(), spread, settings)
        iter = 0
        while not es.stop():
            if shrunk_check:
                sigmas.append(es.sigma)
                if iter > sigma_conv:
                    if all(sigmas[-(i + 1)] < sigmas[-(i + 2)] for i in range(sigma_conv - 1)):
                        print(f"C3:STATUS:Shrunk cloud for {sigma_conv} steps. " "Switching to gradients.")
                        break

            if sigma_check:
                if es.sigma < stop_sigma:
                    print("C3:STATUS:Goal sigma reached. Stopping CMA.")
                    break

            samples = es.ask()
            if init_point and iter == 0:
                samples.insert(0, x_init)
                print("C3:STATUS:Adding initial point to CMA sample.")
            solutions = []
            if batch_noise:
                error = np.random.randn() * noise
            for sample in samples:
                goal = self._setParametersAndMeasure(sample)
                if noise:
                    error = np.random.randn() * noise
                if batch_noise or noise:
                    goal = goal + error
                solutions.append(float(goal))
            es.tell(samples, solutions)
            es.disp()

            iter += 1
            if self._callback is not None:
                self._callback(samples)

        if self._logger:
            self._logger.stop(es.result_pretty())

        return OptimisationResult(
            status=self.__determineTerminationStatus(es.result.stop()),
            value=es.result.fbest,
            iterations=es.result.iterations,
            rawResult=es.result,
        )

    def _setParametersAndMeasure(self, values) -> np.ndarray:
        """Update the parameter values and return the measurement result.

        Internal callback.

        Parameters
        ----------
        values : numpy.ndarray
            Values for the update of the parameters.

        Returns
        -------
        numpy.ndarray
            Returns the inverse of the fidelity.

        """
        log = []
        params = self._optimisables.get_all_parameters()
        for index, val in enumerate(np.split(values, self._opt_idxs[:-1])):
            params[index].set_reduced_value(val)
            log.append(params[index])
        infid = 1 - self._measure.measureNormalised()

        if self._logger:
            self._logger.log(log, infid)
        return infid

    def __determineTerminationStatus(self, conditions: dict) -> int:
        """Determine the optimisation termination status.

        Determines the success or failure of the optimisation depending on the
        termination conditions dict of the CMAEvolutionStrategy.

        Parameters
        ----------
        conditions : dict
            The dictionary from the CMAEvolutionStrategy.stop().

        Returns
        -------
        int
            One of the constants in OptimisationResult.

        """
        if any(
            (key in conditions)
            for key in [
                "ftarget",
                "tolfun",
                "tolfunhist",
                "tolfunrel",
                "tolfacupx",
                "tolx",
            ]
        ):
            return OptimisationResult.STATUS_SUCCESS
        elif any((key in conditions) for key in ["maxfevals", "maxiter", "timeout"]):
            return OptimisationResult.STATUS_FAILED
        else:
            # not decidable
            return OptimisationResult.STATUS_FINISHED
