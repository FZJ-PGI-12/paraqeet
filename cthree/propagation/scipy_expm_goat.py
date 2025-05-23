"""Class definition of the Scipy piecewise exponential propagation model.

Uses the GOAT optimisation method.

"""

from functools import partial
from jax import Array, jit
from jax.lax import scan

import numpy as np
import jax.numpy as jnp

from cthree.exceptions import ConfigurationException
from cthree.propagation.scipy_expm import ScipyExpm


class ScipyExpmGOAT(ScipyExpm):
    """Solve EOMs by piecewise exponentation via Scipy using GOAT."""

    def _create_super_state(self, psi: jnp.ndarray, dpsis: jnp.ndarray):
        """Create a state for the system state and also for gradient vectors.

        Parameters
        ----------
        psi : jax.numpy.ndarray
            State of the system.
        dpsis : jax.numpy.ndarray
            Differential of state.

        Returns
        -------
        jax.numpy.ndarray
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
        eom : jax.numpy.ndarray
            Equations of motion in matrix form.
        grads : jax.numpy.ndarray
            Gradients of the system at a particular step.

        Returns
        -------
        jax.numpy.ndarray
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

    def gradient(self, time: np.ndarray) -> tuple[Array, Array]:
        """Solve the GOAT equation for the gradient vector.

        Parameters
        ----------
        time : numpy.ndarray
            Array of timesteps.

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            First dimension is time, second dimension is the parameter.

        """
        if self._initial_state is None:
            raise ConfigurationException("Initial state is not set")

        n_params = self._model.gradient(jnp.array([0.0])).shape[1]
        dim = self._initial_state.shape[0]
        psi = [jnp.array(self._initial_state, dtype=jnp.complex128)]
        dpsis = [[jnp.zeros_like(self._initial_state, dtype=jnp.complex128)] * n_params]

        eom_func = self._model.get_matrix
        grad_func = self._model.gradient

        for ti in range(1, len(time)):
            times, dt = self._construct_times(time, ti)
            psis_t = self._create_super_state(psi[-1], dpsis[-1])

            eom = eom_func(times + dt / 2) * dt
            grads = jnp.array(grad_func(times + dt / 2)) * dt

            psis_t = self._propagate_gradient(n_params, psis_t, eom, grads, jnp.arange(0, len(times), 1))
            psi.append(psis_t[0:dim])
            dpsis.append(jnp.array([psis_t[dim * ii : dim * (ii + 1)] for ii in range(1, n_params + 1)]))
        return jnp.array(psi), jnp.array(dpsis)
