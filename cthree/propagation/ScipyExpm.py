from typing import List

import numpy as np

from cthree.Quantity import Quantity
from cthree.model.Model import Model
from cthree.propagation.Propagation import Propagation
from cthree.QuantumState import QuantumState

import scipy


class ScipyExpm(Propagation):
    """
    Solve the equation of motion by piecewise exponentation with the scipy package.
    """

    def __init__(self, model: Model):
        super().__init__(model)

    def getParameters(self) -> List[Quantity]:
        return []

    def propagate(self, init: QuantumState, time: np.ndarray):
        times = np.linspace(init.getTime(), time, 1001)
        psi_t = init.getVector()
        for t in times:
            equationsOfMotion = self._model.getEquationOfMotion(t)
            dU = scipy.linalg.expm(equationsOfMotion)
            psi_t = dU @ psi_t
        return psi_t
