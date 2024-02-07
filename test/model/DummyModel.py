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

    def getMatrixEOM(self, time: np.ndarray) -> np.ndarray:
        return -1.0j * self._hamiltonian.getMatrix(time)

    def gradient(self, t) -> List[np.ndarray]:
        """
        Compute the gradient of getMatrixEOM.
        """
        return [-1.0j * h for h in self._hamiltonian.gradient(t)]
