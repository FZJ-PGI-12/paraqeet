from typing import List

import numpy as np

from cthree.Quantity import Quantity
from cthree.model.Hamiltonian import Hamiltonian
from cthree.model.Model import Model


class DummyModel(Model):
    def __init__(self, hamiltonian: Hamiltonian):
        super().__init__(hamiltonian)

    def getParameters(self) -> List[Quantity]:
        pass

    def getMatrixEOM(self, t: np.ndarray) -> np.ndarray:
        return -1.0j * self._hamiltonian.getMatrix(t) * (t[1] - t[0])
