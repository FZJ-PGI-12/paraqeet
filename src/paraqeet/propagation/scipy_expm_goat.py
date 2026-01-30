"""Class definition of the Scipy piecewise exponential propagation model.

Uses the GOAT optimization method.

"""

from functools import partial

import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan

from paraqeet.exceptions import ConfigurationException
from paraqeet.propagation.differentiable_propagation import DifferentiablePropagation
from paraqeet.propagation.scipy_expm import ScipyExpm
from paraqeet.quantity import Array


class ScipyExpmGOAT(ScipyExpm, DifferentiablePropagation):
    """Solve EOMs by piecewise exponentation via Scipy using GOAT."""

    def _create_super_state(self, psi: Array, dpsis: Array) -> Array:
        """Create a state for the system state and also for gradient vectors.

        Parameters
        ----------
        psi : Array
            State of the system.
        dpsis : Array
            Differential of state.

        Returns
        -------
        Array
            Returns a super state created from the state and the differential.

        """
        super_state = [psi]
        super_state.extend(dpsis)
        psi_t = jnp.concatenate(super_state)
        return psi_t

    def _create_goat_ham(self, n_params, eom, grads):
        """Create a Hamiltonian for the GOAT optimization method.

        Parameters
        ----------
        n_params: int
            Number of parameters.
        eom: Array
            Equations of motion in matrix form.
        grads: Array
            Gradients of the system at a particular step.

        Returns
        -------
        Array
            Hamiltonian for the GOAT optimization method.

        """
        line = [eom]
        zeros_like_eom = jnp.zeros_like(eom)
        line.extend([zeros_like_eom] * n_params)
        goat_ham_list = [line]
        for ii, dh_dp in enumerate(grads, start=1):
            line = [dh_dp]
            line.extend([zeros_like_eom] * (ii - 1))
            line.append(eom)
            line.extend([zeros_like_eom] * (n_params - ii))
            goat_ham_list.append(line)

        return jnp.block(goat_ham_list)

    @partial(jit, static_argnums=(0, 1))
    def _propagate_gradient(self, n_params, psis_t, eom, grads, steps_arr):
        def propagate_body(psis_t, index):
            goat_ham = self._create_goat_ham(n_params, eom[index], grads[index])
            psis_t = self._propagate_psi(goat_ham, psis_t)
            return psis_t, psis_t

        psis_t, _ = scan(propagate_body, psis_t, steps_arr)
        return psis_t

    def get_value_and_gradient(self, time: Array) -> tuple[Array, Array]:
        """Solve the GOAT equation for the gradient vector.

        Parameters
        ----------
        time: Array
            Array of timesteps.

        Returns
        -------
        tuple[Array, Array]
            First dimension is time, second dimension is the parameter.

        """
        if self._initial_state is None:
            raise ConfigurationException("Initial state is not set")
        if self._model is None:
            raise ConfigurationException("No equation of motion is configured.")
        n_params = self._model.get_value_and_gradient(jnp.array([0.0]))[1].shape[1]
        dim = self._initial_state.shape[0]
        psis = [jnp.array(self._initial_state, dtype=jnp.complex128)]
        dpsis: list[Array] = [jnp.zeros((n_params,) + self._initial_state.shape, dtype=jnp.complex128)]

        grad_func = self._model.get_value_and_gradient

        for ti in range(1, len(time)):
            times, dt = self._construct_times(time, ti)
            psi_t = self._create_super_state(psis[-1], dpsis[-1])

            eom, grads = grad_func(times + dt / 2)
            eom = eom * dt
            grads = jnp.array(grads) * dt

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
