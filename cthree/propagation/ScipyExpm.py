from typing import List

import numpy as np

from cthree.Exceptions import ConfigurationException
from cthree.Quantity import Quantity
from cthree.model.Model import Model
from cthree.propagation.Propagation import Propagation

import scipy


class ScipyExpm(Propagation):
    """
    Solve the equation of motion by piecewise exponentation with the scipy package.
    """

    __res: float
    _initialState: np.ndarray = None

    def __init__(self, model: Model, res: float):
        """Setup propagation method.

        Args:
            model (Model): Provides equation of motion
            res (float): Resolution at which to sample the EOM
        """
        super().__init__(model)
        self.setResolution(res)

    def setInitialState(self, state: np.ndarray):
        self._initialState = state

    def setResolution(self, res):
        self.__res = res

    def getResolution(self):
        return self.__res

    def getParameters(self) -> List[Quantity]:
        """
        Method has no optimizable parameters.

        Returns:
            Empty list
        """
        return []

    def propagate(self, time: np.ndarray):
        """
        Loop over all desired times in time at set resolution.
        """
        if self._initialState is None:
            raise ConfigurationException("Initial state is not set")

        psi = [self._initialState] * len(time)
        eom = self._model.getMatrixEOM
        for ti in range(1, len(time)):
            t0 = time[ti - 1]
            t1 = time[ti]
            steps = int(np.ceil((t1 - t0) * self.__res))
            times = np.linspace(t0, t1, steps, endpoint=False)
            if steps < 2:
                dt = t1 - t0
            else:
                dt = times[1] - times[0]
            psis_t = psi[ti - 1]
            for t in times:
                # Sampling at the center of the interval.
                psis_t = (
                    scipy.linalg.expm(eom(np.reshape(t, (-1, 1)) + dt / 2) * dt)
                    @ psis_t
                )
            psi[ti] = psis_t
        return psi

    def __grad(self, time: np.ndarray):
        """
        Solve the GOAT equation for the gradient vector
        """
        eom = self._model.getMatrixEOM

        n_params = 2

        psi = [self._initialState]
        dpsis = [[np.zeros_like(self._initialState)] * n_params]

        for ti in range(1, len(time)):
            t0 = time[ti - 1]
            t1 = time[ti]
            steps = int(np.ceil((t1 - t0) * self.__res))
            times = np.linspace(t0, t1, steps, endpoint=False)
            if steps < 2:
                dt = t1 - t0
            else:
                dt = times[1] - times[0]
            superState = [psi[-1]]
            superState.extend(dpsis[-1])
            psis_t = np.concatenate(superState)
            for t in times:
                # Sampling at the center of the interval.
                dc_dp_list = self._model.gradient(t + dt / 2)
                dH_dc = self._model._hamiltonian.getDrives()[0]
                dim = dH_dc.shape[0]
                this_h = eom(np.reshape(t, (-1, 1)) + dt / 2)
                h_list = [this_h] * (n_params + 1)
                goat_ham = scipy.linalg.block_diag(*h_list)
                for ii, dc_dp in enumerate(dc_dp_list):
                    goat_ham[
                        np.ix_([dim * (ii + 1), dim * (ii + 2) - 1], [0, dim - 1])
                    ] = (-1.0j * dH_dc * dc_dp)
                psis_t = scipy.linalg.expm(goat_ham * dt) @ psis_t
            psi.append(psis_t[0:dim])
            dpsis.append(
                [psis_t[dim * ii : dim * (ii + 1)] for ii in range(1, n_params + 1)]
            )
        return psi, dpsis

    def gradient(self, time: np.ndarray):
        _, grad = self.__grad(time)
        return grad
