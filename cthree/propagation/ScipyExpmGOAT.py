from functools import partial
from jax import Array, jit
from jax.lax import scan

import numpy as np
import jax.numpy as jnp

from typing import Tuple

from cthree.propagation.ScipyExpm import ScipyExpm


class ScipyExpmGOAT(ScipyExpm):
    """
    Solve the equation of motion by piecewise exponentation with the scipy package.
    """

    _res: float
    _initialState: np.ndarray = None

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
    def _propagateGradient(self, n_params, psis_t, eom, grads, steps_arr):
        def propagateBody(psis_t, index):
            goat_ham = self._createGOATHam(n_params, eom[index], grads[index])
            psis_t = self._propagatePsi(goat_ham, psis_t)
            return psis_t, psis_t

        psis_t, _ = scan(propagateBody, psis_t, steps_arr)
        return psis_t

    def gradient(self, time: np.ndarray) -> Tuple[Array, Array]:
        """Solve the GOAT equation for the gradient vector.

        Parameters
        ----------
        time : np.ndarray
            array of timesteps

        Returns
        -------
        np.ndarray
            first dimension is time, second dimension is the parameter
        """
        n_params = self._model.gradient(jnp.array([0])).shape[1]
        dim = self._initialState.shape[0]
        psi = [jnp.array(self._initialState, dtype=jnp.complex128)]
        dpsis = [[jnp.zeros_like(self._initialState, dtype=jnp.complex128)] * n_params]

        eom_func = self._model.getMatrixEOM
        grad_func = self._model.gradient

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
                jnp.array(
                    [psis_t[dim * ii : dim * (ii + 1)] for ii in range(1, n_params + 1)]
                )
            )
        return jnp.array(psi), jnp.array(dpsis)
