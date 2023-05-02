from typing import List

from cthree.Quantity import Quantity
from cthree.model.Model import Model
from cthree.propagation.Propagation import Propagation

import scipy

class ScipyExpm(Propagation):
    __T: Quantity

    def __init__(self, model: Model, T: Quantity):
        super().__init__(model)
        self.__T = T

    def getParameters(self) -> List[Quantity]:
        return [self.__T]

    def propagate(self):
        equationsOfMotion = self._model.getEquationOfMotion(self.__T)
        return scipy.linalg.expm(equationsOfMotion)