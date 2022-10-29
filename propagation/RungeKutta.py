from typing import List

from Quantity import Quantity
from model.Model import Model
from propagation.Propagation import Propagation


class RungeKutta(Propagation):
    __T: Quantity

    def __init__(self, model: Model, T: Quantity):
        super(Propagation, self).__init__(model)
        self.__T = T

    def getParameters(self) -> List[Quantity]:
        return [self.__T]

    def propagate(self):
        equationsOfMotion = self._model.getEquationOfMotion()
        # do something
        pass

    def __rungeKuttaStep(self):
        pass
