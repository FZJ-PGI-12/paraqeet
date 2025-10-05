"""Optimise a dCRAB pulse by a Scipy gradient based optimiser."""

import jax.numpy as jnp
import numpy as np

from paraqeet.exceptions import ConfigurationException
from paraqeet.measurement.measurement import Measurement
from paraqeet.optimisation_map import OptimisationMap
from paraqeet.optimisers.scipy_optimiser_gradient import ScipyOptimiserGradient
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import dCRABEnvelope
from paraqeet.signal.waveform import DRAGMixer


class dCRABOptimiserGradient(ScipyOptimiserGradient):
    """A dCRAB optimisation method.

    Implements dCRAB optimisation involving super-iterations that adds additional
    optimisation components to the dCRAB envelope and freezes the older parameters.

    *Note - This works with a `dCRABEnvelope` or a `DRAGMixer` with `list[dCRABEnvelope]` as envelopes.*

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
    optimisable: OptimisationMap
        An optimisation map containing all parameters that can be optimised.
    """

    _num_iteration: int = 0
    _super_iteration_every: int
    _max_super_iteration_num: int
    _num_print_every: int
    _old_parameters_dict: dict[int, list[Quantity]]
    _old_opt_idxs_dict: dict[int, list[int]]
    _current_best_params: list[float]
    _current_best_fid: float = 99999

    def __init__(
        self,
        measure: Measurement,
        optimisables: OptimisationMap,
        super_iteration_every: int = 30,
        max_super_iteration_num: int = 10,
        print_every_iteration_num: int = 5,
    ):
        super().__init__(measure, optimisables)
        self._super_iteration_every = super_iteration_every
        self._max_super_iteration_num = max_super_iteration_num
        self._num_print_every = print_every_iteration_num
        self.callback = self._callback_function
        self._old_parameters_dict = {}
        self._old_opt_idxs_dict = {}

    def _dcrab_super_iteration(self) -> None:
        optimisables = list(self._optimisables.get_optimisables())
        relevant_optimisables = []

        dcrab_envs = []
        for opt in optimisables:
            if isinstance(opt, dCRABEnvelope):
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
                raise ConfigurationException(f"Non `dCRABEnvelope` encountered. \n Raised exception {e}")

        # Add new parameters to optmap
        new_coeffs_and_freqs = [env.get_coefficients_and_frequencies() for env in dcrab_envs]
        for env, coeffs_and_freqs in zip(relevant_optimisables, new_coeffs_and_freqs):
            self._optimisables.add(env, coeffs_and_freqs)

        # Get all parameters and update scales for optimistaion
        params = self._optimisables.get_all_parameters()
        self._scales = jnp.array([p.get_scale() for p in params]).flatten()

        # Restart the optimization process
        self.optimise()

    def _callback_function(self, intermediate_result):
        self._num_iteration += 1

        if intermediate_result.fun < self._current_best_fid:
            self._current_best_fid = intermediate_result.fun
            self._current_best_params = intermediate_result.x

        if self._num_iteration // self._super_iteration_every > self._max_super_iteration_num:
            raise StopIteration("Maximum number of super iterations performed.")

        if self._num_iteration % self._num_print_every == 0:
            print(f"Iteration number = {self._num_iteration} \t  Infidelity  = {intermediate_result.fun:.3f}")

        if self._num_iteration >= self._super_iteration_every:
            if self._num_iteration % self._super_iteration_every == 0:
                print(f"==== Starting super-iteration {self._num_iteration // self._super_iteration_every} ====")
                current_params = self._optimisables.get_all_parameters()
                self._old_parameters_dict[len(current_params)] = current_params
                self._old_opt_idxs_dict[len(current_params)] = self._opt_idxs

                # # set to best value first
                # print(f"Setting paramters to current best paramters with fid = {self._current_best_fid}.")
                # self._set_parameters_and_measure(self._current_best_params)

                self._dcrab_super_iteration()

    def _set_parameters_and_measure(self, values) -> float:
        """Update the parameter values and return measurement result.

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
        params = self._optimisables.get_all_parameters()

        # Check for edge case when best result has fewer parameters than the current iteration.
        if len(params) != len(values):
            num_values = len(values)
            print(f"Backtracking some steps and setting some parameters to zero. Got num values = {num_values}")
            if num_values in self._old_parameters_dict:
                # for index, val in enumerate(np.split(values, self._old_opt_idxs_dict[num_values][:-1])):
                #     self._old_parameters_dict[num_values][index].set_reduced_value(val)
                #     log.append(self._old_parameters_dict[num_values][index])

                # for param in params:
                #     if param not in self._old_parameters_dict[num_values]:
                #         param.set_reduced_value(np.array([-1]))

                ids = [id(i) for i in self._old_parameters_dict[num_values]]
                j = 0
                for ii in range(len(params)):
                    if id(params[ii]) in ids:
                        if j >= num_values:
                            raise Exception("j exceeds num_values")
                        params[ii].set_reduced_value(np.array([values[j]]))
                        j += 1
                    else:
                        params[ii].set_reduced_value(np.array([-1]))

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

        fun, grad = self._measure.measure_with_gradient()
        self._grad_cache = grad

        infid = 1.0 - fun
        if self._logger:
            self._logger.log(log, infid)
        return 1 - fun
