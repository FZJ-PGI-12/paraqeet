from abc import abstractmethod
from typing import List

from cthree.Optimisable import Optimisable
from cthree.Quantity import Quantity
from cthree.model.Hamiltonian import Hamiltonian

import numpy as np


class Model(Optimisable):
    """
    Represents the equation of motion for a given Hamiltonian. Implementations can for example be the Schrödinger
    equation for a closed system, Lindbladian for an open system, or Hamilton's equations for a classical system.
    """

    _hamiltonian: Hamiltonian

    def __init__(self, hamiltonian: Hamiltonian):
        self._hamiltonian = hamiltonian

    @abstractmethod
    def getParameters(self) -> List[Quantity]:
        raise NotImplementedError()

    def getEquationOfMotion(self, time: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Returns the right-hand side of the equations of motion. The format depends on the implementation and could for
        example be a state vector or a matrix. Default implementation assumes a homogeneous ODE with matrix operator
        given by self.getMatrixEOM().

        Args:
            time (np.ndarray): any one-dimensional vector of timestamps

        Returns:
            np.ndarray: the right-hand side of the equation of motion at each time stamp
        """
        return self.getMatrixEOM(time) @ state

    @abstractmethod
    def getMatrixEOM(self, time: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def gradient(self):
        """
        Passthrough
        """
        return self._hamiltonian.gradient()
