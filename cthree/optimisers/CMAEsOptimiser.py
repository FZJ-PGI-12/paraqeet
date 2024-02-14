from typing import Callable, Dict
from cthree.FileLogger import Logger
from cthree.OptimisationMap import OptimisationMap
from cthree.measurement.Measurement import Measurement
from cthree.optimisers.Optimiser import Optimiser
import numpy as np
import cma.evolution_strategy as cma


class CMAEsOptimiser(Optimiser):
    """
    Wrapper for the pycma implementation of CMA-Es. See also:

    http://cma.gforge.inria.fr/apidocs-pycma/

    Parameters
    ----------
    _options : dict
        Options of pycma and the following custom options.

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
    """

    _options: Dict

    def __init__(
        self, measure: Measurement, optimisables: OptimisationMap, logger: Logger = None
    ):
        super().__init__(measure, optimisables, logger)
        self._options = {
            "noise": 0,
            "batch_noise": 0,
            "init_point": False,
            "spread": 0.1,
            "bounds": [-1.0, 1],
        }

    def getOptions(self) -> Dict:
        return self._options

    def setOptions(self, opts):
        self._options.update(opts)

    def setCallback(self, cbfun: Callable) -> None:
        self._callback = cbfun

    def optimise(self) -> cma.CMAEvolutionStrategyResult:
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
        self._optimisables.registerParamsWithOptimisables()

        x_init = []
        for qty in self._optimisables.getAllParameters():
            x_init.append(qty.getReducedValue())

        es = cma.CMAEvolutionStrategy(
            np.concatenate(x_init).flatten(), spread, settings
        )
        iter = 0
        while not es.stop():
            if shrunk_check:
                sigmas.append(es.sigma)
                if iter > sigma_conv:
                    if all(
                        sigmas[-(i + 1)] < sigmas[-(i + 2)]
                        for i in range(sigma_conv - 1)
                    ):
                        print(
                            f"C3:STATUS:Shrunk cloud for {sigma_conv} steps. "
                            "Switching to gradients."
                        )
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
            self._callback(samples)

        if self._logger:
            self._logger.stop(es.result_pretty())

        return es.result

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
