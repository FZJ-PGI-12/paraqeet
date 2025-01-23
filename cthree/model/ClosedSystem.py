"""Class definition of a closed model."""

from cthree.Quantity import Quantity
from cthree.model.Hamiltonian import Hamiltonian
from cthree.model.EquationOfMotion import EquationOfMotion

import numpy as np


class ClosedSystem(EquationOfMotion):
    """Model of a closed physical system, defined by a Hamiltonian.

    Its dynamics is given by the Schrödinger equation.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Matrix representation of a Hamiltonian.

    """

    def __init__(self, hamiltonian: Hamiltonian):
        super().__init__(hamiltonian)

    def get_parameters(self) -> list[Quantity]:
        """Get a list of optimisable parameters.

        Returns
        -------
        List[Quantity]
            List of optimisable parameters of the system.

        """
        return self._hamiltonian.get_parameters()

    def get_matrix(self, time: np.ndarray) -> np.ndarray:
        """Get the matrix equations of motion.

        Computes the right hand side of the Schrödinger equation
        without multiplying the state.
        Used for unitary solvers.

        Parameters
        ----------
        time : numpy.ndarray
            Vector of time samples.

        Returns
        -------
        numpy.ndarray
            RHS with dimension [t, n, n]  with 't' as time
            and 'n' as Hilbert space dimension.

        """
        return -1.0j * self._hamiltonian.get_matrix(time)

    def gradient(self, t) -> np.ndarray:
        """Compute the gradient of getMatrix.

        Parameters
        ----------
        t : numpy.ndarray
            Vector of time samples.

        Returns
        -------
        numpy.ndarray
            Returns the gradient of getMatrix.

        """
        return -1.0j * self._hamiltonian.gradient(t)
