from typing import List

import numpy as np

from cthree.Quantity import Quantity
from cthree.model.Model import Model
from cthree.propagation.Propagation import Propagation


class Euler(Propagation):
    """
    Simple implementation of first order Euler propagation. Solves the equation of motion d/dt psi(t) = F(psi(t), t) with
    a finite step size d as psi(t+d) = psi(t) + F(psi(t), t). The step size can be variable and is calculated from the
    time array that is passed to the propagate function.
    """
    __initialState: np.ndarray

    def __init__(self, model: Model, initialState: np.ndarray):
        super().__init__(model)
        self.__initialState = initialState

    def getParameters(self) -> List[Quantity]:
        return []

    def propagate(self, time: np.ndarray):
        equationsOfMotion = self._model.getEquationOfMotion(time)
        dt = time[1:] - time[0:-1]
        state = self.__initialState
        for i in range(len(dt)-1):
            state += dt[i] * equationsOfMotion[i]
        return state
