"""Optimise a dCRAB pulse by a Scipy gradient based optimiser."""

from paraqeet.exceptions import ConfigurationException
from paraqeet.measurement.measurement import Measurement
from paraqeet.optimisation_map import OptimisationMap
from paraqeet.optimisers.scipy_optimiser_gradient import ScipyOptimiserGradient
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

        current_coeffs_and_freqs = [env.get_coefficients_and_frequencies() for env in dcrab_envs]

        # Add new parameters to the dCRAB envelope
        for env in dcrab_envs:
            try:
                env.add_new_components(seeds=None)  # Setting the seeds to be None for now.
            except Exception as e:
                raise ConfigurationException(f"Non `dCRABEnvelope` encountered. Raised exception {e}")

        # Append new parameters to optmap
        new_coeffs_and_freqs = [env.get_coefficients_and_frequencies() for env in dcrab_envs]
        for env, coeffs_and_freqs in zip(relevant_optimisables, new_coeffs_and_freqs):
            self._optimisables.append(env, coeffs_and_freqs)

        # Remove current coeffs and freqs from the optmap
        for env, coeffs_and_freqs in zip(relevant_optimisables, current_coeffs_and_freqs):
            self._optimisables.remove(env, coeffs_and_freqs)

        self._optimisables.register_params_with_optimisables()

    def _callback_function(self, intermediate_result):
        if self._num_iteration // self._super_iteration_every > self._max_super_iteration_num:
            raise StopIteration("Maximum number of super iterations performed.")

        if self._num_iteration % self._num_print_every == 0:
            print(f"Iteration number = {self._num_iteration} \t  Infidelity  = {intermediate_result.fun}")

        if self._num_iteration >= self._super_iteration_every:
            if self._num_iteration % self._super_iteration_every == 0:
                print(f"==== Starting super-iteration {self._num_iteration // self._super_iteration_every} ====")
                self._dcrab_super_iteration()

        self._num_iteration += 1
