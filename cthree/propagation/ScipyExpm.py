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
    __init: np.ndarray = None

    def __init__(self, model: Model, res: float):
        """Setup propagation method.

        Args:
            model (Model): Provides equation of motion
            res (float): Resolution at which to sample the EOM
        """
        super().__init__(model)
        self.setResolution(res)

    def setInitialState(self, state: np.ndarray):
        self.__init = state

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
        if self.__init is None:
            raise ConfigurationException('Initial state is not set')

        psi = [self.__init]
        for ti in range(1, len(time)):
            t0 = time[ti - 1]
            t1 = time[ti]
            steps = int(np.ceil((t1 - t0) * self.__res))
            times = np.linspace(t0, t1, steps, endpoint=False)
            dt = times[1] - times[0]
            psi_t = psi[-1]
            for t in times:
                eom = self._model.getMatrixEOM
                # Sampling at the center of the interval.
                dU = scipy.linalg.expm(eom(t + dt / 2))
                psi_t = dU @ psi_t
            psi.append(psi_t)
        return psi
