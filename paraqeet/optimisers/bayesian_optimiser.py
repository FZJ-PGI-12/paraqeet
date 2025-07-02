"""Class definition of the Bayesian Optimiser model."""

from bayes_opt import BayesianOptimization

from paraqeet.measurement.measurement import Measurement
from paraqeet.optimisation_map import OptimisationMap
from paraqeet.optimisers.optimiser import OptimisationResult, Optimiser
from paraqeet.exceptions import ConfigurationException


class BayesianOptimiser(Optimiser):
    """Minimizes the outcome of a measuremnt using Bayesian optimisation.

    This is useful if the evaluation of the measurement is costly.
    This class is mostly a wrapper around the implementing package.

    See Also
    --------
    https://bayesian-optimization.github.io/BayesianOptimization/index.html

    Parameters
    ----------
    measure : Measurement
        The measure to be optimised.
    optimisables : OptimisationMap
        All optimisable parameters.
    initialSamples : int, default=10
        Number of iterations before the explorations starts the exploration
        for the maximum.
    iterations : int, default=100
        Number of iterations where the method attempts to find the maximum
        value.

    """

    _measure: Measurement
    __initial_samples: int
    __iterations: int

    def __init__(
        self,
        measure: Measurement,
        optimisables: OptimisationMap,
        initialSamples=10,
        iterations=100,
    ):
        super().__init__(measure, optimisables)
        self.__initial_samples = initialSamples
        self.__iterations = iterations

    @property
    def initial_samples(self) -> int:
        """Get the initial samples fed to the system."""
        return self.__initial_samples

    @initial_samples.setter
    def initial_samples(self, initialSamples: int) -> None:
        """Set the initial samples for the system."""
        self.__initial_samples = initialSamples

    @property
    def iterations(self) -> int:
        """Get the iterations of the system."""
        return self.__iterations

    @iterations.setter
    def iterations(self, iterations: int) -> None:
        """Set the iterations of the system."""
        self.__iterations = iterations

    def optimise(self) -> OptimisationResult:
        """Optimise the system via the Bayesian optimizer.

        Performs the actual optimisation.

        Returns
        -------
        paraqeet.optimisers.optimiser.OptimisationResult
            Result of optimization via the OptimisationResult object.
            (status, value, iterations and the raw result)

        """
        if self._logger:
            self._logger.start()

        self._optimisables.register_params_with_optimisables()
        params = self._optimisables.get_all_parameters()

        # The optimiser needs a dict of named bounds. We use the parameters'
        # indices in the list as names because the parameters' names might
        # not be unique. The bounds are in the reduced representation
        # because this will be the working range for the optimiser.
        optimiser = BayesianOptimization(
            f=self._set_parameters_and_measure,
            pbounds={str(i): (-1, 1) for i in range(len(params))},
            verbose=2,
            random_state=1,
        )
        optimiser.maximize(init_points=self.__initial_samples, n_iter=self.__iterations)

        # The last measurement is not necessarily the best.
        # We therefore set the optimised parameters to the best value.
        if not optimiser.max:
            raise ConfigurationException("BaysianOptimization has no max field.")

        bestValues = optimiser.max["params"]
        for i, param in enumerate(params):
            param.set_reduced_value(bestValues[str(i)])

        # Use the actual names and the non-reduced values
        # for the return value
        result = {params[i].get_name(): params[i].get_value() for i in range(len(bestValues))}
        result["fun"] = 1 - optimiser.max["target"]
        if self._logger:
            self._logger.stop(str(result))

        return OptimisationResult(
            status=OptimisationResult.STATUS_FINISHED,
            value=float(result["fun"]),
            iterations=self.__iterations + self.__initial_samples,
            raw_result=optimiser.max,
        )

    def _set_parameters_and_measure(self, **kwargs) -> float:
        """Update the parameter values and returns the measurement result.

        Internal callback.

        Parameters
        ----------
        **kwargs
            A dict mapping parameter names to their values.

        Returns
        -------
        Array
            Returns the fidelity after setting the parameters.

        """
        log = []
        params = self._optimisables.get_all_parameters()
        for i, param in enumerate(params):
            param.set_reduced_value(kwargs[str(i)])
            log.append(params[i])

        fidelity = self._measure.measure_normalised_scalar()

        if self._logger:
            self._logger.log(log, fidelity)
        return fidelity
