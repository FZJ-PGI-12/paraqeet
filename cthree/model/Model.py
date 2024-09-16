"""Class definition of the optimisable model."""

from abc import abstractmethod

from cthree.Optimisable import Optimisable
from cthree.Quantity import Quantity
from cthree.model.Hamiltonian import Hamiltonian

import numpy as np


class Model(Optimisable):
    """Represents the equation of motion for a given Hamiltonian.

    Implementations can for example be the Schrödinger equation for a
    closed system, Lindbladian for an open system, or Hamilton's equations
    for a classical system.

    Parameters
    ----------
    hamiltonian : cthree.model.Hamiltonian
        Matrix representation of a Hamiltonian.

    """

    _hamiltonian: Hamiltonian

    def __init__(self, hamiltonian: Hamiltonian):
        self._hamiltonian = hamiltonian

    @abstractmethod
    def getParameters(self) -> list[Quantity]:
        """Abstract method to get parameters of the model.

        Returns
        -------
        List[cthree.Quantity]
            Returns the list of parameters as Quantities.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    def getEquationOfMotion(
        self, time: np.ndarray, state: np.ndarray
    ) -> np.ndarray:
        """Return the right-hand side of the equations of motion.

        The format depends on the implementation and could for example
        be a state vector or a matrix. Default implementation assumes a
        homogeneous ODE with matrix operator given by self.getMatrixEOM().

        Parameters
        ----------
        time : numpy.ndarray
            Any one-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            The right-hand side of the equation of motion at each time stamp.

        """
        return self.getMatrixEOM(time) @ state

    @abstractmethod
    def getMatrixEOM(self, time: np.ndarray) -> np.ndarray:
        """Abstract method to get MatrixEOM.

        Parameters
        ----------
        time : numpy.ndarray
            Any one-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Returns matrix equations of motion.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    @abstractmethod
    def gradient(self, t):
        """Implement the gradient of either getEquationOfMotion or getMatrixEOM.

        Parameters
        ----------
        t
            Any one-dimensional vector of timestamps.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()
