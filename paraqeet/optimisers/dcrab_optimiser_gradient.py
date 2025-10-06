"""Optimise a dCRAB pulse by a Scipy gradient based optimiser."""

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from paraqeet.exceptions import ConfigurationException, IncompatibleOptimisationMap
from paraqeet.measurement.measurement import Measurement
from paraqeet.optimisation_map import OptimisationMap
from paraqeet.optimisers.optimiser import OptimisationResult, Optimiser
from paraqeet.optimisers.scipy_optimiser_gradient import ScipyOptimiserGradient
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import DCRABEnvelope
from paraqeet.signal.waveform import DRAGMixer

import warnings

warnings.simplefilter("once")


# Replace the default formatwarning
def custom_formatwarning(msg, *args, **kwargs):
    """Prettier warning statements."""
    return str(msg) + "\n"


warnings.formatwarning = custom_formatwarning  # type: ignore


class DCRABOptimiserGradient(ScipyOptimiserGradient):
    """A dCRAB optimisation method.

    Implements dCRAB optimisation involving super-iterations that adds additional
    optimisation components to the dCRAB envelope and freezes the older parameters.

    *Note - This works with a `DCRABEnvelope` or a `DRAGMixer` with `list[DCRABEnvelope]` as envelopes.*

    _num_iteration: int
        Current iteration number.
    _super_iteration_every: int
        Number of iterations after which one super-iteration is performed. Defaults to 30.
    _max_super_iteration_num: int
        Maximum number of super-iterations to perform. Defaults to 10.
    _num_print_every: int
        Print every this many iterations the current optimisation value. Defaults to 5.
    _old_parameters_dict : dict[int, list[Quantity]]
        Store the parameters of the previous super-iteration in a dictonary labelled by the number of parameters.

    Parameters
    ----------
    measure: Measurement
        Measurement class that measures the observable to be maximised.
    optimisation_map: OptimisationMap
        An optimisation map containing all parameters that can be optimised.
    """

    _num_iteration: int = 0
    _super_iteration_every: int
    _max_super_iteration_num: int
    _num_print_every: int
    _old_parameters_dict: dict[int, list[Quantity]]
    _old_opt_idxs_dict: dict[int, list[int]]
    _best_params: list[float]
    _best_fid: float = 99999
    _previous_fid: float = 99999
    _super_iteration_since: int = 0
    _super_iteration_num: int = 0
    _fallback_optimisation: Optimiser | None

    def __init__(
        self,
        measure: Measurement,
        optimisation_map: OptimisationMap,
        super_iteration_every: int = 30,
        max_super_iteration_num: int = 10,
        print_every_iteration_num: int = 5,
        fallback_optimisation: Optimiser | None = None,
    ):
        super().__init__(measure, optimisation_map)
        self._super_iteration_every = super_iteration_every
        self._max_super_iteration_num = max_super_iteration_num
        self._num_print_every = print_every_iteration_num
        self.callback = self._callback_function
        self._old_parameters_dict = {}
        self._old_opt_idxs_dict = {}
        self._fallback_optimisation = fallback_optimisation

    def _dcrab_super_iteration(self) -> None:
        optimisables = list(self._optimisation_map.get_optimisables())
        relevant_optimisables = []

        dcrab_envs = []
        for opt in optimisables:
            if isinstance(opt, DCRABEnvelope):
                dcrab_envs.append(opt)
                relevant_optimisables.append(opt)
            elif isinstance(opt, DRAGMixer):
                dcrab_envs.extend(opt.get_envelopes())  # type:ignore
                relevant_optimisables.append(opt)  # type:ignore

        # Add new parameters to the dCRAB envelope
        for env in dcrab_envs:
            try:
                env.add_new_components(seeds=None)  # Setting the seeds to be None for now.
            except Exception as e:
                raise ConfigurationException(f"Non `DCRABEnvelope` encountered. \n Raised exception {e}")

        # Add new parameters to optmap
        new_coeffs_and_freqs = [env.get_coefficients_and_frequencies() for env in dcrab_envs]
        for env, coeffs_and_freqs in zip(relevant_optimisables, new_coeffs_and_freqs):
            params = [env.get_parameters()[0]] + coeffs_and_freqs  # Add pulse amplitude to optimisation
            if isinstance(env, DRAGMixer):
                params.append(env.get_parameters()[-1])  # Add pulse delta to optimisation if it is a DRAG tone.
            self._optimisation_map.add(env, params)

        # Get all parameters and update scales for optimistaion
        params = self._optimisation_map.get_all_parameters()
        self._scales = jnp.array([p.get_scale() for p in params]).flatten()
        print(f"* Current no. of parameters = {len(params)}")

        # Restart the optimization process
        self._build_optimisable_index_list()
        self._optimisation_map.register_params_with_optimisables()

        init = []
        for qty in self._optimisation_map.get_all_parameters():
            init.append(qty.get_reduced_value())

        self._minimize_infidelity(init)

    def _callback_function(self, intermediate_result):
        if self._num_iteration % self._num_print_every == 0:
            print(f"Iteration number = {self._num_iteration} \t  Infidelity  = {intermediate_result.fun:.3f}")

        self._num_iteration += 1
        self._super_iteration_since += 1

        if intermediate_result.fun < self._best_fid:
            self._best_fid = intermediate_result.fun
            self._best_params = intermediate_result.x

        if self._super_iteration_num > self._max_super_iteration_num:
            raise StopIteration("Maximum number of super iterations performed.")

        if np.abs(intermediate_result.fun - self._previous_fid) < 1e-6:
            self._super_iteration_num += 1

            print("\n")
            print(f"==== Decrease in infidelity less than {1e-6} ====")
            print(f"==== Starting super-iteration {self._super_iteration_num} ====")
            print(f"* Current lowest infidelity = {self._best_fid: .3f}")

            current_params = self._optimisation_map.get_all_parameters()
            self._old_parameters_dict[len(current_params)] = current_params
            self._old_opt_idxs_dict[len(current_params)] = self._opt_idxs

            # Start the dCRAB super-iteration
            self._super_iteration_since = 0
            self._dcrab_super_iteration()

        self._previous_fid = intermediate_result.fun

        if self._super_iteration_since >= self._super_iteration_every:
            if self._num_iteration % self._super_iteration_every == 0:
                self._super_iteration_num += 1

                print("\n")
                print("==== Max iteration before a super-iteration reached ====")
                print(f"==== Starting super-iteration {self._super_iteration_num} ====")
                print(f"*** Current lowest infidelity = {self._best_fid: .3f} ***")

                current_params = self._optimisation_map.get_all_parameters()
                self._old_parameters_dict[len(current_params)] = current_params
                self._old_opt_idxs_dict[len(current_params)] = self._opt_idxs

                # Start the dCRAB super-iteration
                self._dcrab_super_iteration()

    def _set_parameters(self, values):
        """Update the parameter values.

        This method is derived from the `ScipyOptimiserGradient` class and designed
        to catch cases involving mismatch in dimension of parameters.

        Since, in dCRAB, new parameters are added in each super-iteration,
        the previous best result may be one with fewer parameters.
        In that case, all subsequent parameters are set to their minimum value (by setting the reduced value to -1).

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
        params = self._optimisation_map.get_all_parameters()

        # Check for edge case when best result has fewer parameters than the current iteration.
        if len(params) != len(values):
            num_values = len(values)
            warnings.warn(
                f"Backtracking to previous best fidelity or stopping the optimisation. \
                Going to step with {num_values} parameters."
            )
            if num_values in self._old_parameters_dict:
                ids = [id(i) for i in self._old_parameters_dict[num_values]]
                j = 0
                for ii in range(len(params)):
                    if id(params[ii]) in ids:
                        if j >= num_values:
                            raise Exception("Possible duplicate entries in the optmap.")
                        params[ii].set_reduced_value(np.array([values[j]]))
                        j += 1
                    else:
                        params[ii].set_reduced_value(np.array([0]))
                    log.append(params[ii])
            else:
                raise ConfigurationException(
                    f"Got values from optimisation than does not fit in optimisation map. \n \
                    No. of values from optimisation = {num_values} \
                    and no. of objects in optimisation map \
                    (in the differnt super-iterations) = {list(self._old_parameters_dict.keys())}."
                )
        else:
            for index, val in enumerate(np.split(values, self._opt_idxs[:-1])):
                params[index].set_reduced_value(val)
                log.append(params[index])
        return log

    def _set_parameters_and_measure(self, values) -> float:
        """Update the parameter values and return measurement result.

        Parameters
        ----------
        values: Array
            Parameter values for the update.

        Returns
        -------
        Array
            Returns the inverse of the fidelity.

        """
        log = self._set_parameters(values)

        fun, grad = self._measure.measure_with_gradient()
        self._grad_cache = grad

        infid = 1.0 - fun
        if self._logger:
            self._logger.log(log, infid)
        return 1 - fun

    def _minimize_infidelity(self, init):
        result = minimize(
            fun=self._set_parameters_and_measure,
            jac=self._lookup_jac,
            x0=jnp.concatenate(init).flatten(),
            bounds=[(-1, 1)] * self._opt_idxs[-1],
            method=self._method,
            options=self._options,
            callback=self._callback,
        )
        return result

    def optimise(self) -> OptimisationResult:
        """Optimise via the Scipy optimizer gradient model.

        Performs the actual optimisation.

        Returns
        -------
        OptimisationResult
            The result of the optimisation.

        """
        if self._logger:
            self._logger.start()

        self._build_optimisable_index_list()
        self._optimisation_map.register_params_with_optimisables()

        init = []
        for qty in self._optimisation_map.get_all_parameters():
            init.append(qty.get_reduced_value())

        try:
            result = self._minimize_infidelity(init)
        except Exception as e:
            if "_lbfgsb._lbfgsb.setulb: failed to create array from the 7th" + " argument `g`" in str(e):
                raise IncompatibleOptimisationMap(
                    "Number of quantities in optMap differ from number of" + f" gradients computed. \n {e}"
                )
            else:
                raise e

        # Set the parameters to the best params at the end of optimisation
        print("Setting parameters to the best values.")
        self._set_parameters(self._best_params)

        if self._fallback_optimisation is not None:
            print("Performing fallback optimisation")
            self._fallback_optimisation.optimise()

        if self._logger:
            self._logger.stop(str(result))

        return OptimisationResult(
            status=(OptimisationResult.STATUS_SUCCESS if result.success else OptimisationResult.STATUS_FAILED),
            value=result.fun,
            iterations=result.nfev,
            message=result.message,
            raw_result=result,
        )
