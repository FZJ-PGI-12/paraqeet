from typing import List

import numpy as np
from scipy.integrate import RK45

from cthree.Quantity import Quantity
from cthree.model.Model import Model
from cthree.propagation.StatePropagation import StatePropagation


class RungeKutta(StatePropagation):
    """
    Uses scipy's Runge Kutta implementation for propagating a state vector or density matrix.
    """

    def __init__(self, model: Model):
        super().__init__(model)

    def getParameters(self) -> List[Quantity]:
        return []

    def propagate(self, time: np.ndarray):
        if len(time) < 2:
            raise ValueError('Runge-Kutta propagation needs at least two time steps')

        RK45(
            fun=self._model.getEquationOfMotion,
            t0=time[0],
            y0=self._initialState,
            t_bound=time[-1],
            first_step=time[1] - time[0],
            vectorized=True,
        )
