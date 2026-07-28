"""Class definition of the Bayesian Optimizer model."""

from collections.abc import Callable

import jax.numpy as jnp
from bayes_opt import BayesianOptimization

from paraqeet.exceptions import ConfigurationException
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.optimizer import OptimizationResult, Optimizer
from paraqeet.quantity import Array, Float


class BayesianOptimizer(Optimizer):
    """Minimize the outcome of a measurement using Bayesian optimization :cite:p:`shahriari2016taking`.

    This is useful if the evaluation of the measurement is costly.
    This class is mostly a wrapper around the implementing package, see
    `BayesianOptimization
    <https://bayesian-optimization.github.io/BayesianOptimization/index.html>`_.
    """

    _measure_func: Callable[[Array], Float]
    _initial_samples: int
    _iterations: int

    def __init__(
        self,
        measure_func: Callable[[Array], Float],
        optimization_map: OptimizationMap,
        initial_samples: int = 10,
        iterations: int = 100,
    ) -> None:
        """
        Args:
            measure_func: Function implementing measurement of observables
                to be minimized.
            optimization_map: All optimizable parameters via the optimization
                map.
            initial_samples: Number of initial samples explored before the
                search for the maximum starts.
            iterations: Number of iterations where the method attempts to
                find the maximum value.
        """
        super().__init__(measure_func, optimization_map)
        self._initial_samples = initial_samples
        self._iterations = iterations

    @property
    def initial_samples(self) -> int:
        """Get the initial samples fed to the system."""
        return self._initial_samples

    @initial_samples.setter
    def initial_samples(self, initial_samples: int) -> None:
        """Set the initial samples for the system."""
        self._initial_samples = initial_samples

    @property
    def iterations(self) -> int:
        """Get the iterations of the system."""
        return self._iterations

    @iterations.setter
    def iterations(self, iterations: int) -> None:
        """Set the iterations of the system."""
        self._iterations = iterations

    def optimize(self, times: Array | float) -> OptimizationResult:
        """Optimize the system via the Bayesian optimizer.

        Performs the actual optimization.

        Note:
            If input ``times`` is a float, then the start time of propagation
            is implicitly assumed to be zero. For an array of times, the first
            time point is the start time.

        Args:
            times: Array of times or a float (assumed start time zero).

        Returns:
            Result of optimization (status, value, iterations, and the raw
            result).
        """
        self._times = jnp.array([0.0, times]) if isinstance(times, float) else times

        if self._logger:
            self._logger.start()

        self._optimization_map.register_params_with_optimizables()
        params = self._optimization_map.get_all_parameters()

        # The optimizer needs a dict of named bounds. We use the parameters'
        # indices in the list as names because the parameters' names might
        # not be unique. The bounds are in the reduced representation
        # because this will be the working range for the optimizer.
        optimizer = BayesianOptimization(
            f=self._set_parameters_and_measure,
            pbounds={str(i): (-1, 1) for i in range(len(params))},
            verbose=2,
            random_state=1,
        )
        optimizer.maximize(init_points=self._initial_samples, n_iter=self._iterations)

        # The last measurement is not necessarily the best.
        # We therefore set the optimized parameters to the best value.
        if not optimizer.max:
            raise ConfigurationException("BayesianOptimization has no max field.")

        best_values = optimizer.max["params"]
        for i, param in enumerate(params):
            param.set_reduced_value(best_values[str(i)])

        # Use the actual names and the non-reduced values
        # for the return value
        result = {params[i].get_name(): params[i].get_value() for i in range(len(best_values))}
        result["fun"] = 1 - optimizer.max["target"]
        if self._logger:
            self._logger.stop(str(result))

        return OptimizationResult(
            status=OptimizationResult.STATUS_FINISHED,
            value=float(result["fun"]),
            iterations=self._iterations + self._initial_samples,
            raw_result=optimizer.max,
        )

    def _set_parameters_and_measure(self, **kwargs: float) -> float:
        """Update the parameter values and return the measurement result.

        Internal callback.

        Args:
            **kwargs: A dict mapping parameter names to their values.

        Returns:
            The fidelity after setting the parameters.
        """
        log = []
        params = self._optimization_map.get_all_parameters()
        for i, param in enumerate(params):
            param.set_reduced_value(kwargs[str(i)])
            log.append(params[i])

        fidelity = self._measure_func(self._times)

        if self._logger:
            self._logger.log(log, float(fidelity))
        return float(fidelity)
