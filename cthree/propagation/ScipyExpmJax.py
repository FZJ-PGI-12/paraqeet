from typing import List, Tuple
import numpy as np

import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan, dynamic_update_slice
from jax.scipy.linalg import block_diag
from jax.typing import ArrayLike
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

    def propagate(self, time: np.ndarray) -> List[ArrayLike]:
        """
        Overwrite the `propagte` implementation.
        """
        psi = [jnp.array(self._initialState, dtype=jnp.complex128)]
        eom_func = vmap(self._model.getMatrixEOM)
        for ti in range(1, len(time)):
            times, dt, steps = self._constructTimes(time, ti)
            psis_t = psi[ti - 1]
            eom = vmap(eom_func)(jnp.reshape(times, (-1, 1)) + dt / 2) * dt
            psis_t = self._propagateInTime(psis_t, eom, jnp.arange(0, steps, 1))
            psi.append(psis_t)
        return psi

    def _createSuperState(self, psi, dpsis):
        """
        Create a state with `psi` for the system state and dpsis for gradient vectors.
        """
        superState = [psi]
        superState.extend(dpsis)
        psi_t = jnp.concatenate(superState)
        return psi_t

    def _createGOATHam(self, n_params, dim, EOM_grad, dt, t):
        this_h = self._model.getMatrixEOM(jnp.reshape(t, (-1, 1)) + dt / 2)
        h_list = jnp.repeat(this_h[jnp.newaxis, :, :], n_params + 1, axis=0)
        goat_ham = block_diag(*h_list)
        for ii, dH_dp in enumerate(EOM_grad):
            goat_ham = dynamic_update_slice(goat_ham, dH_dp, (dim * (ii + 1), 0))
        return goat_ham

    def _propagteGradient(self, n_params, dim, psis_t, times, dt):
        def propagateBody(psis_t, t):
            EOM_grad = self._model.gradient(t + dt / 2)
            goat_ham = self._createGOATHam(n_params, dim, EOM_grad, dt, t)
            psis_t = self._propagatePsi(goat_ham * dt, psis_t)
            return psis_t, psis_t

        psis_t, _ = scan(propagateBody, psis_t, jnp.array(times))
        return psis_t

    def gradient(self, time: np.ndarray) -> Tuple[ArrayLike, List[List[ArrayLike]]]:
        """
        Solve the GOAT equation for propagating the gradient vectors.
        Compatible with Jax and Jit.

        Args:
            time (np.ndarray): Array of timesteps

        Returns:
            Tuple[ArrayLike, List[List[ArrayLike]]]: Return the propagated states and gradient vectors.
        """

        n_params = len(self._model.gradient(0))
        dim = self._initialState.shape[0]
        psi = [jnp.array(self._initialState, dtype=jnp.complex128)]
        dpsis = [[jnp.zeros_like(self._initialState, dtype=jnp.complex128)] * n_params]

        for ti in range(1, len(time)):
            times, dt = self._constructTimes(time, ti)
            psis_t = self._createSuperState(psi[-1], dpsis[-1])
            psis_t = self._propagteGradient(n_params, dim, psis_t, times, dt)
            psi.append(psis_t[0:dim])
            dpsis.append(
                [psis_t[dim * ii : dim * (ii + 1)] for ii in range(1, n_params + 1)]
            )
        return psi, dpsis
