from typing import List

from cthree.Quantity import Quantity
from cthree.model.Hamiltonian import Hamiltonian
from cthree.model.Model import Model

import numpy as np


class ClosedModel(Model):
    """
    Model of a closed physical system, defined by a Hamiltonian. Its dynamics given by the Schrödinger equation.
    """

    def __init__(self, hamiltonian: Hamiltonian):
        super().__init__(hamiltonian)

    def getParameters(self) -> List[Quantity]:
        """
        Optimizable parameters.

        Returns:
            List[Quantity]: List of optimizable parameters.
        """
        return self._hamiltonian.getParameters()

    def getMatrixEOM(self, time: np.ndarray) -> np.ndarray:
        """
        Computes the right hand side of the Schrödinger equation without multiplying the state. Used for unitary
        solvers.

        Args:
            time (np.ndarray): Vector of time samples

        Returns:
            np.ndarray: RHS with dimension [t, n, n]  with t: time, n: hilbert space
        """
        return -1.0j * self._hamiltonian.getMatrix(time)

    def gradient(self, t) -> List[np.ndarray]:
        """
        Compute the gradient of getMatrixEOM.
        """
        return [-1.0j * h for h in self._hamiltonian.gradient(t)]
