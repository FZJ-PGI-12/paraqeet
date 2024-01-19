from typing import List
import numpy as np

import jax.numpy as jnp
from jax import jit
from jax.lax import scan
from jax.typing import ArrayLike
from functools import partial

from cthree.model.Model import Model
from cthree.propagation.ScipyExpmGOAT import ScipyExpmGOAT


class ScipyExpmJax(ScipyExpmGOAT):
    """
    Jax based implementation of ScipyExmpGOAT.
    Allows faster propagation and gradient computation while using AutoGrad.
    """

    def __init__(self, model: Model, res: float):
        super().__init__(model, res)

    @partial(jit, static_argnums=(0,))
    def _propagateInTime(self, psis_t, times, dt):
        def propagateBody(psis_t, t):
            eom = self._model.getMatrixEOM(jnp.reshape(t, (-1, 1)) + dt / 2) * dt
            psis_t = self._propagatePsi(eom, psis_t)
            return psis_t, psis_t

        psis_t, _ = scan(propagateBody, psis_t, jnp.array(times))

        return psis_t

    def propagate(self, time: np.ndarray) -> List[ArrayLike]:
        """
        Overwrite the `propagte` implementation.
        Has inputs to `_propagatePsi` as jnp.array.
        """
        psi = [jnp.array(self._initialState, dtype=jnp.complex128)]
        for ti in range(1, len(time)):
            times, dt = self._constructTimes(time, ti)
            psis_t = psi[ti - 1]
            psis_t = self._propagateInTime(psis_t, times, dt)
            psi.append(psis_t)
        return psi
