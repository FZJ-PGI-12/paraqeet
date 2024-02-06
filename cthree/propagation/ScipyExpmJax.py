from typing import List, Tuple
import numpy as np

import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan, dynamic_update_slice
from jax.scipy.linalg import block_diag
from jax import Array
from functools import partial

from cthree.model.Model import Model
from cthree.propagation.ScipyExpm import ScipyExpm


class ScipyExpmJax(ScipyExpm):
    """
    Jax based implementation of ScipyExmpGOAT.
    Allows faster propagation and gradient computation while using AutoGrad.
    """

    def __init__(self, model: Model, res: float):
        super().__init__(model, res)

    @partial(jit, static_argnums=(0,))
    def _propagateInTime(self, psis_t, eom, steps_arr):
        """
        Propagate from `time[ti] to time[ti+1]`.
        JIT compiled and uses `jax.lax.scan` to avoid compilation overhead.
        """

        def propagateBody(psis_t, index):
            psis_t = self._propagatePsi(eom[index], psis_t)
            return psis_t, psis_t

        psis_t, _ = scan(propagateBody, psis_t, steps_arr)

        return psis_t

    def propagate(self, time: np.ndarray) -> Array:
        """
        Overwrite the `propagte` implementation.
        """
        psi = [jnp.array(self._initialState, dtype=jnp.complex128)]
        eom_func = self._model.getMatrixEOM
        for ti in range(1, len(time)):
            times, dt = self._constructTimes(time, ti)
            psis_t = psi[ti - 1]
            eom = eom_func(times + dt / 2) * dt
            psis_t = self._propagateInTime(psis_t, eom, jnp.arange(0, len(times), 1))
            psi.append(psis_t)
        return jnp.array(psi)

    def _createSuperState(self, psi, dpsis):
        """
        Create a state with `psi` for the system state and dpsis for gradient vectors.
        """
        superState = [psi]
        superState.extend(dpsis)
        psi_t = jnp.concatenate(superState)
        return psi_t

    def _createGOATHam(self, n_params, eom, grads):
        line = [eom]
        zeros_like_eom = jnp.zeros_like(eom)
        line.extend([zeros_like_eom]*n_params)
        goat_ham_list = [line]
        for ii, dH_dp in enumerate(grads, start=1):
            line = [dH_dp]
            line.extend([zeros_like_eom]*(ii-1))
            line.append(eom)
            line.extend([zeros_like_eom] *(n_params - ii))
            goat_ham_list.append(line)

        return jnp.block(goat_ham_list)


    @partial(jit, static_argnums=(0, 1))
    def _propagateGradient(self, n_params, psis_t, eom, grads, steps_arr):
        def propagateBody(psis_t, index):
            goat_ham = self._createGOATHam(n_params, eom[index], grads[:, index])
            psis_t = self._propagatePsi(goat_ham, psis_t)
            return psis_t, psis_t

        psis_t, _ = scan(propagateBody, psis_t, steps_arr)
        return psis_t

    def gradient(self, time: np.ndarray) -> Tuple[Array, Array]:
        """
        Solve the GOAT equation for propagating the gradient vectors.
        Compatible with Jax and Jit.

        Args:
            time (np.ndarray): Array of timesteps

        Returns:
            Tuple[ArrayLike, ArrayLike]: Return the propagated states and gradient vectors.
        """

        n_params = len(self._model.gradient(0))
        dim = self._initialState.shape[0]
        psi = [jnp.array(self._initialState, dtype=jnp.complex128)]
        dpsis = [[jnp.zeros_like(self._initialState, dtype=jnp.complex128)] * n_params]

        eom_func = vmap(self._model.getMatrixEOM)
        grad_func = vmap(self._model.gradient)

        for ti in range(1, len(time)):
            times, dt = self._constructTimes(time, ti)
            psis_t = self._createSuperState(psi[-1], dpsis[-1])

            eom = eom_func(times + dt / 2) * dt
            grads = jnp.array(grad_func(times + dt / 2)) * dt

            psis_t = self._propagateGradient(
                n_params, psis_t, eom, grads, jnp.arange(0, len(times), 1)
            )
            psi.append(psis_t[0:dim])
            dpsis.append(
                jnp.array([psis_t[dim * ii : dim * (ii + 1)] for ii in range(1, n_params + 1)])
            )
        return jnp.array(psi), jnp.array(dpsis)
