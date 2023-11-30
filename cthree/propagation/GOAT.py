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
    _TIME_SCALE: float

    def __init__(self, model: Model, initialTimeStep: float | None = None):
        """
        :param initialTimeStep: Optional initial time step for the adaptive time steps in RK45.
        """
        super().__init__(model)
        self.__initialTimeStep = initialTimeStep
        self._TIME_SCALE = 1e-9

    def setInitialState(self, state: np.ndarray):
        """
        Sets the initial state for the propagation.

        :param state:
        :return:
        """
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
        if self._initialState is None:
            raise ConfigurationException("Initial state is not set")

        if len(time) < 2:
            raise ValueError("Runge-Kutta propagation needs at least two time steps")

        TIME_SCALE = self._TIME_SCALE  # ns

        def callback(time, state):
            column_state = np.reshape(state, (-1, 1))
            return np.reshape(
                TIME_SCALE
                * self._model.getEquationOfMotion(
                    np.array([time]) * TIME_SCALE, column_state
                ),
                (-1,),
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
                t0=time[ti - 1] / TIME_SCALE,
                y0=np.reshape(states[-1], (-1,)),
                t_bound=time[ti] / TIME_SCALE,
                first_step=dt / TIME_SCALE,
                vectorized=False,
            )

            while integrator.status == "running":
                integrator.step()
            states.append(np.reshape(integrator.y, (-1, 1)))
        return states

    def __grad(self, time: np.ndarray):
        """
        Solve the GOAT equation for the gradient vector
        """

        TIME_SCALE = self._TIME_SCALE  # ns

        def callback(time, state):
            column_state = np.reshape(state, (-1, 1))
            return np.reshape(
                TIME_SCALE
                * self._model.getEquationOfMotion(
                    np.array([time]) * TIME_SCALE, column_state
                ),
                (-1,),
            )

        def coEom(time, dpsi_dp, psi):
            time = np.reshape(time, (-1,)) * TIME_SCALE
            dH_dp = self._model._hamiltonian.getDrives()[0]
            return (
                -1j * (dH_dp @ psi) + self._model.getMatrixEOM(time) @ dpsi_dp
            ) * TIME_SCALE

        psi = [np.reshape(self._initialState, (-1,))]
        dpsi = [np.zeros((psi[0].size, 1), dtype=np.complex128)]

        for ti in range(1, len(time)):
            dt = self.__initialTimeStep
            if dt is None or dt > time[ti] - time[ti - 1]:
                dt = (time[ti] - time[ti - 1]) / 5

            integrator = RK45(
                fun=callback,
                t0=time[ti - 1] / TIME_SCALE,
                y0=psi[-1],
                t_bound=time[ti] / TIME_SCALE,
                first_step=dt / TIME_SCALE,
                vectorized=False,
            )
            dpsi_t = 0
            dpsi_t += dpsi[-1]
            psi_t = dpsi[-1]
            while integrator.status == "running":
                integrator.step()
                dt = integrator.step_size
                dpsi_t += (coEom(integrator.t - dt, dpsi_t, psi_t)) * dt
                psi_t = np.reshape(integrator.y, (-1, 1))
            dpsi.append(dpsi_t)
            psi.append(integrator.y)
        return dpsi

    def gradient(self, time: np.ndarray):
        dpsi_dc = self.__grad(time)
        dt = 0.001e-9
        ts = np.arange(time[0], time[-1], dt)
        dc_dp_list = self._model.gradient(ts)
        dpsi_dp = [dpsi_dc[-1] * np.sum(dc_dp) * dt for dc_dp in dc_dp_list]
        return dpsi_dp
