"""Class definition of the Scipy piecewise exponential propagation model."""

from functools import partial

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

    Solve the equation of motion by piecewise exponentation with the
    Scipy package.

    Parameters
    ----------
    model: Model
        Represents the equation of motion for a given Hamiltonian.
    res: float
        Resolution at which to sample the EOM.

    """

    @partial(jit, static_argnums=(0,))
    def _propagate_in_time(self, psis_t, eom, steps_arr):
        """Propagate the system in time.

        Iteratively propagate state/states (psis_t) according
        to the equation of motion (eom). The eom is exponentiated using
        `jax.scipy.linalg.expm` to compute the propagators.
        The iterations use `jax.lax.scan` to avoid compilation overhead.

        Parameters
        ----------
        psis_t: Array
            State/states at time 't'.
        eom: Array
            Equation of motion for a list of times.
        steps_arr: Array
            Array from 0 to the length of the List of time, in steps of 1
            representing the iteration index.

        Returns
        -------
        Array
            Returns the evolved state.

        """

        def propagate_body(psis_t, index):
            psis_t = ScipyExpm._propagate_psi(eom[index], psis_t)
            return psis_t, psis_t

        psis_t, _ = scan(propagate_body, psis_t, steps_arr)

        return psis_t

    @staticmethod
    @jit
    def _propagate_psi(eom_matrix, psis_t):
        """Propagate the state/states (psis_t).

        Parameters
        ----------
        eom_matrix : Array
            The equations of motion matrix.
        psis_t : Array
            State/states at time 't'.

        Returns
        -------
        Array
            Returns the evolved state.

        """
        return expm(eom_matrix) @ psis_t

    def propagate(self, time: Array) -> Array:
        """Return the solution of the equations of motion.

        Loop over all desired times in time at set resolution.

        Parameters
        ----------
        time: Array
            Any one-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the solution of the equations of motion.

        Raises
        ------
        ConfigurationException
            If the initial state is not set.

        """
        if len(time) < 2:
            raise ValueError("ScipyExpm.propagate needs at least two time points.")

        psis = [self._initial_state]

        for ti in range(1, len(time)):
            times, dt = construct_times(time, ti, self._resolution)
            psis_t = psis[ti - 1]
            eom = self._eom_func(times + dt / 2) * dt
            psis_t = self._propagate_in_time(psis_t, eom, jnp.arange(0, len(times), 1))
            psis.append(psis_t)

        psis_arr = jnp.array(psis)
        return psis_arr
