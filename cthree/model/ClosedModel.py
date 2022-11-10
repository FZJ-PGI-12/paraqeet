from typing import List

from cthree.Quantity import Quantity
from cthree.model.Hamiltonian import Hamiltonian
from cthree.model.Model import Model

import numpy as np


class ClosedModel(Model):
    def __init__(self, hamiltonian: Hamiltonian):
        super().__init__(hamiltonian)

    def getParameters(self) -> List[Quantity]:
        return []

    def getEquationOfMotion(self) -> np.ndarray:
        return -1.0j * self._hamiltonian.getMatrix()
