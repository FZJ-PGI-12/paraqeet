"""Class definition of the Scipy piecewise exponential propagation model."""

from functools import partial

import jax.numpy as jnp
from paraqeet.quantity import Array
from jax import jit, vmap
from jax.lax import scan
from jax.scipy.linalg import expm

from paraqeet.exceptions import ConfigurationException
from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.propagation.state_propagation import StatePropagation
from paraqeet.quantity import Quantity

import jax

jax.config.update("jax_enable_x64", True)


class ScipyExpm(StatePropagation):
    """Piecewise matrix exponential propagation system.

    Solve the equation of motion by piecewise exponentation with the
    Scipy package.

    Parameters
    ----------
    model : Model
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
        List[Quantity]
            Returns an empty list.

        """
        return []

    def _construct_times(self, time, ti):
        """Construct one-dimensional vector of time.

        In specified resolution at a snapshot.

        Parameters
        ----------
        time : Array
            Array of timesteps.
        ti : int
            Snapshot of the time at a current step

        Returns
        -------
        Array
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

    def set_initial_state(self, state):
        """Set initial state."""
        # Verify if `model.ode_propagation` is set to `False`.
        # ode_propgation returns hamiltonian and collapse operators separately.
        eom = self._model.get_matrix(jnp.array([0]))
        if len(eom) == 2:
            raise ConfigurationException("Please set `model.ode_propagation` to `False` for this propagation method.")

        # For open system convert Density Matrix to Vectorized form.
        if self.is_open:
            try:
                if len(state.shape) == 1:  # An (n,) array
                    state = jnp.reshape(state, (-1, 1))
                # Compare the shapes of inital state with the generator of time translation
                dim_generator = eom.shape[1]
                # Comparing dim -2 as 0 can be batch dimension
                if state.shape[-2] == jnp.sqrt(dim_generator):
                    # check if it is a square matrix. Check the last 2 dimensions are equal.
                    if state.shape[-1] == state.shape[-2]:
                        # This is a density matrix
                        state = self._convert_dm_to_vec(state)
            except Exception as e:
                raise ConfigurationException(
                    f"Obtained a state vector of shape {state.shape} as initial state. "
                    + "For open system propagation expected a density matrix or vectorized density matrix "
                    + "as the initial state.\n"
                    + f"Raised exception: `{e}`"
                )
        self._initial_state = jnp.array(state, dtype=jnp.complex128)

    @staticmethod
    def _convert_dm_to_vec(state_dm: Array) -> jnp.ndarray:
        """Helper function to convert a density matrix to vectorized form."""
        return jnp.reshape(jnp.transpose(state_dm), (-1, 1))

    @staticmethod
    def _convert_vec_to_dm(state_vec: Array, dim: int) -> jnp.ndarray:
        """Helper function to convert a Vectorized density matrix to matrix form."""
        return jnp.transpose(jnp.reshape(state_vec, (dim, dim)))

    @partial(jit, static_argnums=(0,))
    def _propagate_in_time(self, psis_t, eom, steps_arr):
        """Propagate the system in time.

        Iteratively propagate state/states (psis_t) according
        to the equation of motion (eom). The eom is exponentiated using
        `jax.scipy.linalg.expm` to compute the propagators.
        The iterations use `jax.lax.scan` to avoid compilation overhead.

        Parameters
        ----------
        psis_t : chtree.quantity.Array
            State/states at time 't'.
        eom : chtree.quantity.Array
            Equation of motion for a list of times.
        steps_arr : chtree.quantity.Array
            Array from 0 to the length of the List of time, in steps of 1
            representing the iteration index.

        Returns
        -------
        chtree.quantity.Array
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
        eom_matrix : chtree.quantity.Array
            The equations of motion matrix.
        psis_t : chtree.quantity.Array
            State/states at time 't'.

        Returns
        -------
        chtree.quantity.Array
            Returns the evolved state.

        """
        return expm(eom_matrix) @ psis_t

    def propagate(self, time: Array) -> Array:
        """Return the solution of the equations of motion.

        Loop over all desired times in time at set resolution.

        Parameters
        ----------
        time : Array
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
        if self._initial_state is None:
            raise ConfigurationException("Initial state is not set")

        init_state = jnp.array(self._initial_state, dtype=jnp.complex128)

        if self._model is not None:
            eom_func = self._model.get_matrix
        else:
            raise ConfigurationException("No equation of motion is configured.")
        psis = [init_state]

        for ti in range(1, len(time)):
            times, dt = self._construct_times(time, ti)
            psis_t = psis[ti - 1]
            eom = eom_func(times + dt / 2) * dt
            psis_t = self._propagate_in_time(psis_t, eom, jnp.arange(0, len(times), 1))
            psis.append(psis_t)

        psis_arr = jnp.array(psis)
        # if open system convert back the vectorized density matrices to matrix shape
        if self.is_open:
            dim = int(jnp.sqrt(eom.shape[-1]))
            psis_arr = vmap(self._convert_vec_to_dm, in_axes=(0, None))(psis_arr, dim)
        return psis_arr
