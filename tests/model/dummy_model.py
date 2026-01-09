"""Class definition of the Dummy model for testing."""

from paraqeet.model.differentiable_hamiltonian import DifferentiableHamiltonian
from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.quantity import Array, Quantity


class DummyEquationsOfMotion(EquationOfMotion):
    """Dummy model class to construct derived model classes.

    Parameters
    ----------
    hamiltonian: Hamiltonian
        Class object for a matrix representation of a Hamiltonian.
    """

    def __init__(self, hamiltonian: DifferentiableHamiltonian):
        super().__init__(hamiltonian)

    def get_parameters(self) -> list[Quantity]:
        """Get parameters of the system.

        Returns
        -------
        list[Quantity]
            List of parameters of the system.

        """
        return []

    def get_value(self, t: Array) -> Array:
        """Get the matrix representation of the equations of motion.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the matrix equations of motion.

        """
        return -1.0j * self._hamiltonian.get_value(t)

    def get_gradient(self, t) -> Array:
        """Compute the gradient of get_value.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        """
        return -1.0j * self._hamiltonian.gradient(t)
