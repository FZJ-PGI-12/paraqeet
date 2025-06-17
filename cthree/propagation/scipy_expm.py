"""Class definition of the Scipy piecewise exponential propagation model."""

from functools import partial

import jax.numpy as jnp
from jax import Array, jit
from jax.lax import scan
from jax.scipy.linalg import expm

from cthree.exceptions import ConfigurationException
from cthree.model.equation_of_motion import EquationOfMotion
from cthree.propagation.state_propagation import StatePropagation
from cthree.quantity import Quantity

import jax

jax.config.update("jax_enable_x64", True)


class ScipyExpm(StatePropagation):
    """Piecewise matrix exponential propagation system.

    Solve the equation of motion by piecewise exponentation with the
    Scipy package.

    Parameters
    ----------
    model : cthree.model.Model
        Represents the equation of motion for a given Hamiltonian.
    res : float
        Resolution at which to sample the EOM.

    """

    _res: float
    _initial_state: Array | None = None

    def __init__(self, model: EquationOfMotion, res: float):
        super().__init__(model)
        self.resolution = res

    @property
    def resolution(self) -> float:
        """Get the resolution of the system."""
        return self._res

    @resolution.setter
    def resolution(self, res: float):
        """Set the resolution of the propagation."""
        self._res = res

    def get_parameters(self) -> list[Quantity]:
        """Get a list of optimisable parameters of the system.

        Note: Method has no optimisable parameters.

        Returns
        -------
        List[cthree.Quantity]
            Returns an empty list.

        """
        return []

    def _construct_times(self, time, ti):
        """Construct one-dimensional vector of time.

        In specified resolution at a snapshot.

        Parameters
        ----------
        time : numpy.ndarray
            Array of timesteps.
        ti : int
            Snapshot of the time at a current step

        Returns
        -------
        numpy.ndarray
            Array of timestamps in specified resolution.
        int
            Difference in time step.

        """
        t0 = time[ti - 1]
        t1 = time[ti]
        steps = int(jnp.ceil((t1 - t0) * self._res))
        times = jnp.linspace(t0, t1, steps, endpoint=False)
        if steps < 2:
            dt = t1 - t0
        else:
            dt = times[1] - times[0]
        return times, dt

    @partial(jit, static_argnums=(0,))
    def _propagate_in_time(self, psis_t, eom, steps_arr):
        """Propagate the system in time.

        Iteratively propagate state/states (psis_t) according
        to the equation of motion (eom). The eom is exponentiated using
        `jax.scipy.linalg.expm` to compute the propagators.
        The iterations use `jax.lax.scan` to avoid compilation overhead.

        Parameters
        ----------
        psis_t : jax.numpy.ndarray
            State/states at time 't'.
        eom : jax.numpy.ndarray
            Equation of motion for a list of times.
        steps_arr : jax.numpy.ndarray
            Array from 0 to the length of the List of time, in steps of 1
            representing the iteration index.

        Returns
        -------
        jax.Array
            Returns the evolved state.

        """

        def propagate_body(psis_t, index):
            psis_t = self._propagate_psi(eom[index], psis_t)
            return psis_t, psis_t

        psis_t, _ = scan(propagate_body, psis_t, steps_arr)

        return psis_t

    @staticmethod
    @jit
    def _propagate_psi(eom_matrix, psis_t):
        """Propagate the state/states (psis_t).

        Parameters
        ----------
        eom_matrix : jax.numpy.ndarray
            The equations of motion matrix.
        psis_t : jax.numpy.ndarray
            State/states at time 't'.

        Returns
        -------
        jax.Array
            Returns the evolved state.

        """
        return expm(eom_matrix) @ psis_t

    def propagate(self, time: Array) -> Array:
        """Return the solution of the equations of motion.

        Loop over all desired times in time at set resolution.

        Parameters
        ----------
        time : numpy.ndarray
            Any one-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the solution of the equations of motion.

        Raises
        ------
        cthree.Exceptions.ConfigurationException
            If the initial state is not set.

        """
        if self._initial_state is None:
            raise ConfigurationException("Initial state is not set")

        psi = [jnp.array(self._initial_state, dtype=jnp.complex128)]
        eom_func = self._model.get_matrix
        for ti in range(1, len(time)):
            times, dt = self._construct_times(time, ti)
            psis_t = psi[ti - 1]
            eom = eom_func(times + dt / 2) * dt
            psis_t = self._propagate_in_time(psis_t, eom, jnp.arange(0, len(times), 1))
            psi.append(psis_t)
        return jnp.array(psi)
