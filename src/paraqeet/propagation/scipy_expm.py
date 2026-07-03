"""Class definition of the Scipy piecewise exponential propagation model."""

from typing import override

import jax
import jax.numpy as jnp
from jax import jit
from jax.lax import scan
from jax.scipy.linalg import expm

from paraqeet.propagation.propagation import StatePropagation
from paraqeet.propagation.utils import construct_times
from paraqeet.quantity import Array

jax.config.update("jax_enable_x64", True)


class ScipyExpm(StatePropagation):
    """Piecewise matrix exponential propagation system.

    Solve the equation of motion by piecewise exponentiation with the
    Scipy package.

    """

    @staticmethod
    @jit
    def _propagate_in_time(psis_t: Array, eom: Array, steps_arr: Array):
        """Propagate the system in time.

        Iteratively propagate state/states (psis_t) according
        to the equation of motion (eom). The eom is exponentiated using
        `jax.scipy.linalg.expm` to compute the propagators.
        The iterations use `jax.lax.scan` to avoid compilation overhead.

        Args:
            psis_t: State/states at time 't'.
            eom: Equation of motion for a list of times.
            steps_arr: Array from 0 to the length of the List of times, in steps
                of 1 representing the iteration index.

        Returns:
            The evolved state.

        """

        def propagate_body(psis_t, index):
            psis_t = ScipyExpm._propagate_psi(eom[index], psis_t)
            return psis_t, psis_t

        psis_t, _ = scan(propagate_body, psis_t, steps_arr)

        return psis_t

    @staticmethod
    @jit
    def _propagate_psi(eom_matrix: Array, psis_t: Array):
        """Propagate the state/states (psis_t).

        Args:
            eom_matrix: The equations of motion matrix.
            psis_t: State/states at time 't'.

        Returns:
            Array: Returns the evolved state.

        """
        return expm(eom_matrix) @ psis_t

    @override
    def propagate(self, times: Array) -> Array:
        """Return the solution of the equations of motion.

        Loop over all desired times in time at set resolution.

        Args:
            times: Array of times.

        Returns:
            The solution of the equations of motion.

        Raises:
            ConfigurationException: If the initial state is not set.

        """
        if len(times) < 2:
            raise ValueError("ScipyExpm.propagate needs at least two time points.")

        psis = [self._initial_state]

        for ti in range(1, len(times)):
            step_times, dt = construct_times(times, ti, self._resolution)
            psis_t = psis[ti - 1]
            eom = self._eom_func(step_times + dt / 2) * dt
            psis_t = self._propagate_in_time(psis_t, eom, jnp.arange(0, len(step_times), 1))
            psis.append(psis_t)

        psis_arr = jnp.array(psis)
        return psis_arr
