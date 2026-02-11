"""Class definition of the Scipy piecewise exponential propagation model."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan
from jax.scipy.linalg import expm

from paraqeet.exceptions import ConfigurationException
from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array

jax.config.update("jax_enable_x64", True)


class ScipyExpm(Propagation):
    """Piecewise matrix exponential propagation system.

    Solve the equation of motion by piecewise exponentiation with the
    Scipy package.

    Parameters
    ----------
    model: Model
        Represents the equation of motion for a given Hamiltonian.
    res: float
        Resolution at which to sample the EOM.

    """

    _resolution: float
    _initial_state: Array | None = None

    def __init__(self, model: EquationOfMotion, resolution: float):
        super().__init__(model, resolution)
        self.resolution = resolution

    def set_initial_state(self, state):
        """Set initial state."""
        # Verify if `model.ode_propagation` is set to `False`.
        # ode_propgation returns hamiltonian and collapse operators separately.
        eom = self._model.get_value(jnp.array([0.0]))
        dim_generator = eom.shape[1]

        if len(eom) == 2:
            raise ConfigurationException("Please set `model.ode_propagation` to `False` for this propagation method.")

        state = self._check_and_fix_state_shape(state, dim_generator)
        self._initial_state = jnp.array(state, dtype=jnp.complex128)

    def _check_and_fix_state_shape(self, state, dim_generator):
        # For closed system check if the initial state has the right dimensions.
        if not self.is_open:
            if len(state.shape) == 1:  # (n,) array
                state = jnp.reshape(state, (-1, 1))

            if state.shape[-2] != dim_generator:
                raise ConfigurationException(
                    f"Obtained a state vector of shape {state.shape} as initial state. "
                    + f"Expected a shape of dimensions `{(dim_generator, 1)}`"
                    + "or an array of initial states of the above dimension."
                )

        # For open system convert Density Matrix to Vectorized form.
        else:
            try:
                if len(state.shape) == 1:  # An (n,) array
                    state = jnp.reshape(state, (-1, 1))

                # Check whether it is a density matrix or vectorized density matrix (or an array of those).
                # Checking if it is a density matrix. Comparing dim (-2) as (0) can be batch dimension
                if state.shape[-2] == jnp.sqrt(dim_generator):
                    # check if it is a square matrix. Check the last 2 dimensions are equal.
                    if state.shape[-1] == state.shape[-2]:
                        # This is a density matrix. Convert to vectorized form.
                        state = ScipyExpm._convert_dm_to_vec(state, int(jnp.sqrt(dim_generator)))

                    # check if it is a list of vectorized density matrices
                    elif state.shape[-1] == dim_generator:
                        # This is an edge case with batch dimension = n, where n is Hilbert space dimension
                        state = jnp.expand_dims(state, -1)

                    else:
                        raise ConfigurationException(
                            "Initial state neither a density matrix nor a vectorized density matrix.\n"
                            + "For a list of vectorized density matrices expected shape is (m, n^2, 1)"
                            + " where m is the batch dimension, n is the Hilbert space dimension."
                        )

                # Not a density matrix (or list). Check if it is a vectorized density matrix.
                elif state.shape[-2] == dim_generator:
                    # vectorized density matrix or a list of vectorized density matrix
                    if state.shape[-1] == state.shape[-2]:
                        # list of vectorized density matrices
                        # This is an edge case: batch dimension = n^2, where n is Hilbert space dimension
                        state = jnp.expand_dims(state, -1)

                    elif state.shape[-1] == 1:
                        # normal vectorized density matrix
                        state = state

                    else:
                        raise ConfigurationException(
                            "Initial state neither a single or list of vectorized density matrix.\n"
                            + "For a list of vectorized density matrices expected shape is (m, n^2, 1)"
                            + " where m is the batch dimension, n is the Hilbert space dimension."
                        )

            except Exception as e:
                raise ConfigurationException(
                    f"Obtained a state vector of shape {state.shape} as initial state. "
                    + "For open system propagation expected a density matrix or vectorized density matrix "
                    + "as the initial state.\n"
                    + f"Raised exception: `{e}`"
                )

        return state

    @staticmethod
    def _convert_dm_to_vec(state_dm: Array, dim: int) -> jnp.ndarray:
        """Helper function to convert a density matrix to vectorized form."""
        vec = jnp.reshape(jnp.transpose(state_dm), (-1, dim**2, 1))
        if vec.shape[0] == 1:
            vec = jnp.squeeze(vec, axis=0)
        return vec

    @staticmethod
    def _convert_vec_to_dm(state_vec: Array, dim: int) -> jnp.ndarray:
        """Helper function to convert a Vectorized density matrix to matrix form."""
        dm = jnp.reshape(state_vec, (-1, dim, dim))
        if dm.shape[0] == 1:
            dm = jnp.squeeze(dm, axis=0)
        return jnp.transpose(dm)

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

        if self._initial_state is None:
            raise ConfigurationException("Initial state is not set")

        init_state = jnp.array(self._initial_state, dtype=jnp.complex128)

        if self._model is not None:
            eom_func = self._model.get_value
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
            psis_arr = vmap(ScipyExpm._convert_vec_to_dm, in_axes=(0, None))(psis_arr, dim)
        return psis_arr

    # TODO: implement get_collapseops method
