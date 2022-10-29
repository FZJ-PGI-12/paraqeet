from typing import List

from Quantity import Quantity
from model.Model import Model


class ClosedModel(Model):
    def getParameters(self) -> List[Quantity]:
        return []

    def getEquationOfMotion(self):
        return -1.0j * self._hamiltonian.getMatrix()
