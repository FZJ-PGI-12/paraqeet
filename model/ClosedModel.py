from typing import List

from Quantity import Quantity
from model.Hamiltonian import Hamiltonian
from model.Model import Model


class ClosedModel(Model):
    def __init__(self, hamiltonian: Hamiltonian):
        super(Model, self).__init__(hamiltonian)

    def getParameters(self) -> List[Quantity]:
        return []

    def getEquationOfMotion(self):
        return -1.0j * self._hamiltonian.getMatrix()
