import numpy as np
from bayes_opt import BayesianOptimization

from cthree.OptimisationMap import OptimisationMap
from cthree.Optimiser import Optimiser
from cthree.measurement.Measurement import Measurement


class BayesianOptimiser(Optimiser):
    """
    Minimizes the outcome of a measuremnt using Bayesian optimisation. This is useful if the evaluation of the
    measurement is costly. This class is mostly a wrapper around the implementing package.

    :param measure: the measure to be optimised
    :param optimisables: all optimisable parameters
    :param initialSamples: Number of iterations before the explorations starts the exploration for the maximum.
    :param iterations: Number of iterations where the method attempts to find the maximum value
    """

    _measure: Measurement
    __initialSamples: int
    __iterations: int

    def __init__(self, measure: Measurement, optimisables: OptimisationMap, initialSamples=10, iterations=100):
        super().__init__(measure, optimisables)
        self.__initialSamples = initialSamples
        self.__iterations = iterations

    def getInitialSamples(self) -> int:
        return self.__initialSamples

    def setInitialSamples(self, initialSamples: int):
        self.__initialSamples = initialSamples

    def getIterations(self) -> int:
        return self.__iterations

    def setIterations(self, iterations: int):
        self.__iterations = iterations

    def optimise(self) -> dict:
        if self._logger:
            self._logger.start()

        self._optimisables.registerParamsWithOptimisables()
        params = self._optimisables.getAllParameters()

        # The optimiser needs a dict of named bounds. We use the parameters' indices in the list as names because the
        # parameters' names might not be unique. The bounds are in the reduced representation because this will be the
        # working range for the optimiser.
        optimiser = BayesianOptimization(
            f=self._setParametersAndMeasure,
            pbounds={str(i): (-1, 1) for i in range(len(params))},
            verbose=2,
            random_state=1,
        )
        optimiser.maximize(init_points=self.__initialSamples, n_iter=self.__iterations)

        # The last measurement is not necessarily the best. We therefore set the optimised parameters to the best value.
        bestValues = optimiser.max['params']
        for i, param in enumerate(params):
            param.setReducedValue(bestValues[str(i)])

        # Use the actual names and the non-reduced values for the return value
        result = {params[i].getName(): params[i].getValue() for i in range(len(bestValues))}
        if self._logger:
            self._logger.stop(str(result))

        return result

    def _setParametersAndMeasure(self, **kwargs) -> np.ndarray:
        """
        Updates the parameter values and returns the measurement result. Internal callback.

        :param kwargs: a dict mapping parameter names to their values
        """
        log = []
        params = self._optimisables.getAllParameters()
        for i, param in enumerate(params):
            param.setReducedValue(kwargs[str(i)])
            log.append(params[i])

        fidelity = self._measure.measureNormalised()

        if self._logger:
            self._logger.log(log, fidelity)
        return fidelity
