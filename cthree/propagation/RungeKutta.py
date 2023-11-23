from typing import List

import numpy as np
from scipy.integrate import RK45

from cthree.Exceptions import ConfigurationException
from cthree.Quantity import Quantity
from cthree.model.Model import Model
from cthree.propagation.StatePropagation import StatePropagation


class RungeKutta(StatePropagation):
    """
    Uses scipy's Runge Kutta implementation for propagating a state vector or density matrix.
    """

    __initialTimeStep: float

    def __init__(self, model: Model, initialTimeStep: float | None = None):
        """
        :param initialTimeStep: Optional initial time step for the adaptive time steps in RK45.
        """
        super().__init__(model)
        self.__initialTimeStep = initialTimeStep

    def getParameters(self) -> List[Quantity]:
        return []

    def setInitialState(self, state: np.ndarray):
        """
        Sets the initial state for the propagation. Subclasses can access the state in the _initialState field.

        :param state:
        :return:
        """
        self._initialState = np.reshape(state, (-1,))

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
