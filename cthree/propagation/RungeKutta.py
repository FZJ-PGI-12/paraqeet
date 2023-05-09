from typing import List

import numpy as np

from cthree.Quantity import Quantity
from cthree.model.Model import Model
from cthree.propagation.Propagation import Propagation


class RungeKutta(Propagation):
    __T: Quantity

    def __init__(self, model: Model, T: Quantity):
        super().__init__(model)
        self.__T = T

    def getParameters(self) -> List[Quantity]:
        return [self.__T]

    def propagate(self, time: np.ndarray):
        equationsOfMotion = self._model.getEquationOfMotion(time)
        # do something
        pass

    def __rungeKuttaStep(self):
        pass
