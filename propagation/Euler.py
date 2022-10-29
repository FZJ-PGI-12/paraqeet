from typing import List

from Quantity import Quantity
from propagation.Propagation import Propagation


class Euler(Propagation):
    __T: Quantity

    def __construct(self, T: Quantity):
        self.__T = T

    def getParameters(self) -> List[Quantity]:
        return [self.__T]

    def propagate(self):
        self._model.getEquationOfMotion()
        pass
