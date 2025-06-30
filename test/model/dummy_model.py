"""Class definition of the Dummy model for testing."""

from paraQeet.quantity import Array

from paraQeet.quantity import Quantity
from paraQeet.model.hamiltonian import Hamiltonian
from paraQeet.model.equation_of_motion import EquationOfMotion


class DummyModel(EquationOfMotion):
    """Dummy model class to construct derived model classes.

    Parameters
    ----------
    hamiltonian : cthree.model.hamiltonian
        Class object for a matrix representation of a Hamiltonian.
    """

    def __init__(self, hamiltonian: Hamiltonian):
        super().__init__(hamiltonian)

    def get_parameters(self) -> list[Quantity]:
        """Get parameters of the system.

        Returns
        -------
        list[cthree.quantity]
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
