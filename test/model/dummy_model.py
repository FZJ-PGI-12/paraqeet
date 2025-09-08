"""Class definition of the Dummy model for testing."""

from paraqeet.quantity import Array

from paraqeet.quantity import Quantity
from paraqeet.model.hamiltonian import Hamiltonian
from paraqeet.model.equation_of_motion import EquationOfMotion


class DummyEquationsOfMotion(EquationOfMotion):
    """Dummy model class to construct derived model classes.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Class object for a matrix representation of a Hamiltonian.
    """

    def __init__(self, hamiltonian: Hamiltonian):
        super().__init__(hamiltonian)

    def get_parameters(self) -> list[Quantity]:
        """Get parameters of the system.

        Returns
        -------
        list[Quantity]
            List of parameters of the system.

        """
        return []

    def get_matrix(self, time: Array) -> Array:
        """Get the matrix representation of the equations of motion.

        Parameters
        ----------
        time : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the matrix equations of motion.

        """
        return -1.0j * self._hamiltonian.get_matrix(time)

    def gradient(self, t) -> Array:
        """Compute the gradient of getMatrixEOM.

        Parameters
        ----------
        t : Array
            One-dimensional vector of timestamps.

        """
        return -1.0j * self._hamiltonian.gradient(t)
