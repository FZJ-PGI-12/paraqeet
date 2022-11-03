from typing import List

from cthree.Quantity import Quantity
from cthree.model.Hamiltonian import Hamiltonian
from cthree.model.Model import Model


class ClosedModel(Model):
    def __init__(self, hamiltonian: Hamiltonian):
        super().__init__(hamiltonian)

    def getParameters(self) -> List[Quantity]:
        return []

    def getEquationOfMotion(self):
        return -1.0j * self._hamiltonian.getMatrix()
