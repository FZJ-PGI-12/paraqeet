from functools import partial
from typing import List

import numpy as np
import jax.numpy as jnp

from cthree.Exceptions import ConfigurationException
from cthree.Quantity import Quantity
from cthree.model.Model import Model
from cthree.propagation.StatePropagation import StatePropagation

from jax.scipy.linalg import expm
from jax import Array, jit
from jax.lax import scan


class ScipyExpm(StatePropagation):
    """
    Solve the equation of motion by piecewise exponentation with the scipy package.
    """

    _res: float
    _initialState: np.ndarray = None

    def __init__(self, model: Model, res: float):
        """Setup propagation method.

        Args:
            model (Model): Provides equation of motion
            res (float): Resolution at which to sample the EOM
        """
        super().__init__(model)
        self.setResolution(res)

    def setResolution(self, res: float):
        self._res = res

    def getResolution(self) -> float:
        return self._res

    def getParameters(self) -> List[Quantity]:
        """
        Method has no optimizable parameters.

        Returns:
            Empty list
        """
        return []

    def _constructTimes(self, time, ti):
        t0 = time[ti - 1]
        t1 = time[ti]
        steps = int(np.ceil((t1 - t0) * self._res))
        times = np.linspace(t0, t1, steps, endpoint=False)
        if steps < 2:
            dt = t1 - t0
        else:
            dt = times[1] - times[0]
        return times, dt

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

    @staticmethod
    @jit
    def _propagatePsi(eom_matrix, psis_t):
        return expm(eom_matrix) @ psis_t

    def propagate(self, time: np.ndarray) -> Array:
        """
        Loop over all desired times in time at set resolution.
        """
        if self._initialState is None:
            raise ConfigurationException("Initial state is not set")

        psi = [jnp.array(self._initialState, dtype=jnp.complex128)]
        eom_func = self._model.getMatrixEOM
        for ti in range(1, len(time)):
            times, dt = self._constructTimes(time, ti)
            psis_t = psi[ti - 1]
            eom = eom_func(times + dt / 2) * dt
            psis_t = self._propagateInTime(psis_t, eom, jnp.arange(0, len(times), 1))
            psi.append(psis_t)
        return jnp.array(psi)
