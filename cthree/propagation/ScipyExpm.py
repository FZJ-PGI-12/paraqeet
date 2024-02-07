from typing import List

import numpy as np

from cthree.Exceptions import ConfigurationException
from cthree.Quantity import Quantity
from cthree.model.Model import Model
from cthree.propagation.StatePropagation import StatePropagation

from jax.scipy.linalg import expm
from jax import jit


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

    @staticmethod
    @jit
    def _propagatePsi(eom_matrix, psis_t):
        return expm(eom_matrix) @ psis_t

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

    def propagate(self, time: np.ndarray) -> np.ndarray:
        """
        Loop over all desired times in time at set resolution.
        """
        if self._initialState is None:
            raise ConfigurationException("Initial state is not set")

        psi = np.array([self._initialState] * len(time), dtype=np.complex128)
        eom = self._model.getMatrixEOM
        for ti in range(1, len(time)):
            times, dt = self._constructTimes(time, ti)
            psis_t = psi[ti - 1]
            for t in times:
                # Sampling at the center of the interval.
                # psis_t = expm(eom(np.reshape(t, (-1, 1)) + dt / 2) * dt) @ psis_t
                psis_t = self._propagatePsi(
                    eom(np.reshape(t, (-1, 1)) + dt / 2)[0] * dt, psis_t
                )
            psi[ti] = psis_t
        return psi
