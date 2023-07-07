from typing import List

import numpy as np

from cthree.Quantity import Quantity
from cthree.model.Model import Model
from cthree.propagation.Propagation import Propagation

import scipy


class ScipyExpm(Propagation):
    """
    Solve the equation of motion by piecewise exponentation with the scipy package.
    TODO: Implement resolution setup and looping over timestamps.
    """

    __res: float

    def __init__(self, model: Model, res: float):
        super().__init__(model)
        self.__res = res

    def getParameters(self) -> List[Quantity]:
        return []

    def propagate(self, init: np.ndarray, time: np.ndarray):
        """
        Loop over all desired times in time at reasonable resolution.
        """
        psi = [init]
        for ti in range(1, len(time)):
            t0 = time[ti - 1]
            t1 = time[ti]
            steps = int(np.ceil((t1 - t0) * self.__res))
            times = np.linspace(t0, t1, steps)
            psi_t = psi[-1]
            for t in times:
                equationsOfMotion = self._model.getEquationOfMotion(t)
                dU = scipy.linalg.expm(equationsOfMotion)
                psi_t = dU @ psi_t
            psi.append(psi_t)
        return psi
