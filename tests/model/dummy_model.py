"""Class definition of the Dummy model for testing."""

from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.quantity import Array, Quantity


class DummyEquationsOfMotion(EquationOfMotion):
    """Dummy model class to construct derived model classes.

    Parameters
    ----------
    hamiltonian: Hamiltonian
        Class object for a matrix representation of a Hamiltonian.
    """

    def __init__(self, system):
        super().__init__(system)

    def get_parameters(self) -> list[Quantity]:
        """Get parameters of the system.

        Returns
        -------
        list[Quantity]
            List of parameters of the system.

        """
        return []

    def get_value(self, times: Array) -> Array:
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
        return -1.0j * self._solvable.get_hamiltonian(times)

    def get_value_and_gradient(self, times) -> tuple[Array, Array]:
        """Compute the gradient of get_value.

        Parameters
        ----------
        times: Array
            One-dimensional vector of timestamps.

        """
        ham, grad = self._solvable.get_hamiltonian_and_gradient(times)
        return -1.0j * ham, -1.0j * grad
