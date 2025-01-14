"""Class definition of the Dummy model for testing."""

import numpy as np

from cthree.Quantity import Quantity
from cthree.model.Hamiltonian import Hamiltonian
from cthree.model.EquationOfMotion import EquationOfMotion


class DummyModel(EquationOfMotion):
    """Dummy model class to construct derived model classes.

    Parameters
    ----------
    hamiltonian : cthree.model.Hamiltonian
        Class object for a matrix representation of a Hamiltonian.
    """

    def __init__(self, hamiltonian: Hamiltonian):
        super().__init__(hamiltonian)

    def getParameters(self) -> list[Quantity]:
        """Get parameters of the system.

        Returns
        -------
        list[cthree.Quantity]
            List of parameters of the system.

        """
        pass

    def getMatrix(self, time: np.ndarray) -> np.ndarray:
        """Get the matrix representation of the equations of motion.

        Parameters
        ----------
        time : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Returns the matrix equations of motion.

        """
        return -1.0j * self._hamiltonian.getMatrix(time)

    def gradient(self, t) -> list[np.ndarray]:
        """Compute the gradient of getMatrixEOM.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        """
        return [-1.0j * h for h in self._hamiltonian.gradient(t)]
