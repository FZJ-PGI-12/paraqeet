"""Class definition of the Scipy piecewise exponential propagation model.

Uses the GOAT optimisation method.

"""

from functools import partial
from jax import jit, vmap
from jax.lax import scan

import jax.numpy as jnp

from paraqeet.quantity import Array

from paraqeet.exceptions import ConfigurationException
from paraqeet.propagation.scipy_expm import ScipyExpm


class ScipyExpmGOAT(ScipyExpm):
    """Solve EOMs by piecewise exponentation via Scipy using GOAT."""

    def _create_super_state(self, psi: Array, dpsis: Array) -> Array:
        """Create a state for the system state and also for gradient vectors.

        Parameters
        ----------
        psi : chtree.quantity.Array
            State of the system.
        dpsis : chtree.quantity.Array
            Differential of state.

        Returns
        -------
        chtree.quantity.Array
            Returns a super state created from the state and the differential.

        """
        superState = [psi]
        superState.extend(dpsis)
        psi_t = jnp.concatenate(superState)
        return psi_t

    def _create_GOAT_ham(self, n_params, eom, grads):
        """Create a Hamiltonian for the GOAT optimisation method.

        Parameters
        ----------
        n_params : int
            Number of parameters.
        eom : chtree.quantity.Array
            Equations of motion in matrix form.
        grads : chtree.quantity.Array
            Gradients of the system at a particular step.

        Returns
        -------
        chtree.quantity.Array
            Hamiltonian for the GOAT optimisation method.

        """
        line = [eom]
        zeros_like_eom = jnp.zeros_like(eom)
        line.extend([zeros_like_eom] * n_params)
        goat_ham_list = [line]
        for ii, dH_dp in enumerate(grads, start=1):
            line = [dH_dp]
            line.extend([zeros_like_eom] * (ii - 1))
            line.append(eom)
            line.extend([zeros_like_eom] * (n_params - ii))
            goat_ham_list.append(line)

        return jnp.block(goat_ham_list)

    @partial(jit, static_argnums=(0, 1))
    def _propagate_gradient(self, n_params, psis_t, eom, grads, steps_arr):
        def propagateBody(psis_t, index):
            goat_ham = self._create_GOAT_ham(n_params, eom[index], grads[index])
            psis_t = self._propagate_psi(goat_ham, psis_t)
            return psis_t, psis_t

        psis_t, _ = scan(propagateBody, psis_t, steps_arr)
        return psis_t

    def gradient(self, time: Array) -> tuple[Array, Array]:
        """Solve the GOAT equation for the gradient vector.

        Parameters
        ----------
        time : Array
            Array of timesteps.

        Returns
        -------
        Tuple[Array, Array]
            First dimension is time, second dimension is the parameter.

        """
        if self._initial_state is None:
            raise ConfigurationException("Initial state is not set")
        if self._model is None:
            raise ConfigurationException("No equation of motion is configured.")
        n_params = self._model.gradient(jnp.array([0.0])).shape[1]
        dim = self._initial_state.shape[0]
        psis = [jnp.array(self._initial_state, dtype=jnp.complex128)]
        dpsis: list[Array] = [jnp.zeros((n_params,) + self._initial_state.shape, dtype=jnp.complex128)]

        eom_func = self._model.get_matrix
        grad_func = self._model.gradient

        for ti in range(1, len(time)):
            times, dt = self._construct_times(time, ti)
            psi_t = self._create_super_state(psis[-1], dpsis[-1])

            eom = eom_func(times + dt / 2) * dt
            grads = jnp.array(grad_func(times + dt / 2)) * dt

            psi_t = self._propagate_gradient(n_params, psi_t, eom, grads, jnp.arange(0, len(times), 1))
            psis.append(jnp.array(psi_t[0:dim]))
            dpsis.append(jnp.array([psi_t[dim * ii : dim * (ii + 1)] for ii in range(1, n_params + 1)]))

        psis_arr = jnp.array(psis)
        dpsis_arr = jnp.array(dpsis)

        if self.is_open:
            dim = int(jnp.sqrt(eom.shape[-1]))
            psis_arr = vmap(self._convert_vec_to_dm, in_axes=(0, None))(psis_arr, dim)
            dpsis_arr = vmap(vmap(self._convert_vec_to_dm, in_axes=(0, None)), in_axes=(0, None))(dpsis_arr, dim)
        return psis_arr, dpsis_arr
