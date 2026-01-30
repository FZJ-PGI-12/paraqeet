"""Class definition of the CMA-Es Optimizer model."""

from collections.abc import Callable

import cma.evolution_strategy as cma
import jax.numpy as jnp
import numpy as np

from paraqeet.file_logger import Logger
from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.optimizer import OptimizationResult, Optimizer
from paraqeet.quantity import Array


class CMAEsOptimizer(Optimizer):
    """Wrapper for the pycma implementation of CMA-Es.

    The pycmi implementation has the following custom options for optimization:
    noise: float
        Artificial noise added to a function evaluation.
    init_point: boolean
        Force the use of the initial point in the first generation.
    spread: float
        Adjust the parameter spread of the first generation cloud.
    stop_at_convergence : int
        Custom stopping condition. Stop if the cloud shrunk for this number of
        generations.
    stop_at_sigma: float
        Custom stopping condition. Stop if the cloud shrunk to this standard
        deviation.

    See also: http://cma.gforge.inria.fr/apidocs-pycma/

    Parameters
    ----------
    measure: Measurement
        Represents any observable and the process of measurement itself.
    optimizables: OptimizationMap
        Optimizable interface for all parameters considered in optimization.
    logger: FileLogger | None, default=None
        The file logger object.
    callback
        Callback function for optimization.

    """

    _measure: NormalizableMeasurement
    _options: dict
    _callback: Callable | None

    def __init__(
        self,
        measure: NormalizableMeasurement,
        optimizables: OptimizationMap,
        logger: Logger | None = None,
        callback=None,
    ):
        super().__init__(measure, optimizables, logger)
        self._options = {
            "noise": 0,
            "batch_noise": 0,
            "init_point": False,
            "spread": 0.1,
            "bounds": [-1.0, 1],
        }
        self.callback = callback

    @property
    def options(self) -> dict:
        """Get options from the system."""
        return self._options

    @options.setter
    def options(self, opts) -> None:
        """Set options for the system."""
        self._options.update(opts)

    @property
    def callback(self) -> Callable | None:
        """Returns the current callback function."""
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

    def optimize(self, times: Array | float) -> OptimizationResult:
        """Optimize the system via the CMA-Es optimizer.

        Performs the actual optimization via the following custom options:
        noise: float
            Artificial noise added to a function evaluation.
        init_point: boolean
            Force the use of the initial point in the first generation.
        spread: float
            Adjust the parameter spread of the first generation cloud.
        stop_at_convergence : int
            Custom stopping condition. Stop if the cloud shrunk for this number
            of generations.
        stop_at_sigma: float
            Custom stopping condition. Stop if the cloud shrunk to this
            standard deviation.

        *Note - If input `times` is a float, then the start time of propagation is implicity assumed to be zero.
        For an array of times, the first time point is the start time.*

        Returns
        -------
        OptimizationResult
            Result of optimization via the OptimizationResult object.
            (status, value, iterations and the raw result)

        """
        self._times = jnp.array([0.0, times]) if isinstance(times, float) else times

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

        self._build_optimizable_index_list()
        self._optimizables.register_params_with_optimizables()

        x_init = []
        for qty in self._optimizables.get_all_parameters():
            x_init.append(qty.get_reduced_value())

        es = cma.CMAEvolutionStrategy(np.concatenate(x_init).flatten(), spread, settings)
        iter = 0
        while not es.stop():
            if shrunk_check:
                sigmas.append(es.sigma)
                if iter > sigma_conv:
                    if all(sigmas[-(i + 1)] < sigmas[-(i + 2)] for i in range(sigma_conv - 1)):
                        print(f"ParaQeet:STATUS: Shrunk cloud for {sigma_conv} steps. Switching to gradients.")
                        break

            if sigma_check:
                if es.sigma < stop_sigma:
                    print("ParaQeet:STATUS: Goal sigma reached. Stopping CMA.")
                    break

            samples = es.ask()
            if init_point and iter == 0:
                samples.insert(0, x_init)
                print("ParaQeet:STATUS: Adding initial point to CMA sample.")
            solutions = []
            if batch_noise:
                error = np.random.randn() * noise
            for sample in samples:
                goal = self._set_parameters_and_measure(sample)
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

        return OptimizationResult(
            status=self.__determine_termination_status(es.result.stop()),
            value=es.result.fbest,
            iterations=es.result.iterations,
            raw_result=es.result,
        )

    def _set_parameters_and_measure(self, values) -> float:
        """Update the parameter values and return the measurement result.

        Internal callback.

        Parameters
        ----------
        values: Array
            Values for the update of the parameters.

        Returns
        -------
        Array
            Returns the inverse of the fidelity.

        """
        log = []
        params = self._optimizables.get_all_parameters()
        for index, val in enumerate(np.split(values, self._opt_idxs[:-1])):
            params[index].set_reduced_value(val)
            log.append(params[index])

        infid = 1.0 - self._measure.calculate_normalized_scalar(self._times)

        if self._logger:
            self._logger.log(log, infid)
        return infid

    def __determine_termination_status(self, conditions: dict) -> int:
        """Determine the optimization termination status.

        Determines the success or failure of the optimization depending on the
        termination conditions dict of the CMAEvolutionStrategy.

        Parameters
        ----------
        conditions: dict
            The dictionary from the CMAEvolutionStrategy.stop().

        Returns
        -------
        int
            One of the constants in OptimizationResult.

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
            return OptimizationResult.STATUS_SUCCESS
        elif any((key in conditions) for key in ["maxfevals", "maxiter", "timeout"]):
            return OptimizationResult.STATUS_FAILED
        else:
            # not decidable
            return OptimizationResult.STATUS_FINISHED
