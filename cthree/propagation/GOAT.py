from typing import List

import numpy as np

from cthree.Exceptions import ConfigurationException
from cthree.Quantity import Quantity
from cthree.model.Model import Model
from cthree.propagation.Propagation import Propagation

from scipy.integrate import RK45


class GOAT(Propagation):
    """
    Solve the equation of motion by piecewise exponentation with the scipy package.
    """

    __initialTimeStep: float
    _initialState: np.ndarray
    _timegrid: List[np.ndarray]

    def __init__(self, model: Model, initialTimeStep: float | None = None):
        """
        :param initialTimeStep: Optional initial time step for the adaptive time steps in RK45.
        """
        super().__init__(model)
        self.__initialTimeStep = initialTimeStep

    def setInitialState(self, state: np.ndarray):
        """
        Sets the initial state for the propagation.

        :param state:
        :return:
        """
        self._initialState = np.reshape(state, (-1,))

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
        if self._initialState is None:
            raise ConfigurationException("Initial state is not set")

        if len(time) < 2:
            raise ValueError("Runge-Kutta propagation needs at least two time steps")

        def callback(time, state):
            column_state = np.reshape(state, (-1, 1))
            return np.reshape(
                self._model.getEquationOfMotion(np.array([time]), column_state), (-1,)
            )

        # Since RK45 uses adaptive time steps and does not guarantee to return a state for each time stamp, this
        # function has to iterate over the time steps itself.
        states = [self._initialState]
        for ti in range(1, len(time)):
            dt = self.__initialTimeStep
            if dt is None or dt > time[ti] - time[ti - 1]:
                dt = (time[ti] - time[ti - 1]) / 5

            integrator = RK45(
                fun=callback,
                t0=time[ti - 1],
                y0=states[-1],
                t_bound=time[ti],
                first_step=dt,
                vectorized=False,
            )

            while integrator.status == "running":
                integrator.step()
            states.append(integrator.y)
        return states

    def __grad(self, time: np.ndarray):
        """
        Solve the GOAT equation for the gradient vector
        """
        eom = self._model.getEquationOfMotion

        def coEom(time, dpsi_dp, psi):
            time = np.reshape(time, (-1,))
            dH_dp = self._model._hamiltonian.getDrives()[0]
            return dH_dp @ psi + eom(time, dpsi_dp)

        psi = [self._initialState]
        dpsi = [self._initialState]

        for ti in range(1, len(time)):
            dt = self.__initialTimeStep
            if dt is None or dt > time[ti] - time[ti - 1]:
                dt = (time[ti] - time[ti - 1]) / 5

            integrator = RK45(
                fun=eom,
                t0=time[ti - 1],
                y0=psi[-1],
                t_bound=time[ti],
                first_step=dt,
                vectorized=False,
            )
            dpsi_t = 0
            dpsi_t += dpsi[-1]
            while integrator.status == "running":
                integrator.step()
                dpsi_t += coEom(time[ti], dpsi_t, integrator.y) * integrator.step_size
            dpsi.append(dpsi_t)
        return dpsi

    def gradient(self, time: np.ndarray):
        return self.__grad(time)
