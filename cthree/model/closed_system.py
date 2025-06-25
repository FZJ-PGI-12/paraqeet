"""Class definition of a closed model."""

from cthree.model.equation_of_motion import EquationOfMotion
from cthree.model.hamiltonian import Hamiltonian
from cthree.quantity import Quantity, Array


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

    def get_matrix(self, time: Array) -> Array:
        """Get the matrix equations of motion.

        Computes the right hand side of the Schrödinger equation
        without multiplying the state.
        Used for unitary solvers.

        Parameters
        ----------
        time : Array
            Vector of time samples.

        Returns
        -------
        Array
            RHS with dimension [t, n, n]  with 't' as time
            and 'n' as Hilbert space dimension.

        """
        return -1.0j * self._hamiltonian.get_matrix(time)

    def gradient(self, t) -> Array:
        """Compute the gradient of getMatrix.

        Parameters
        ----------
        t : Array
            Vector of time samples.

        Returns
        -------
        Array
            Returns the gradient of getMatrix.

        """
        return -1.0j * self._hamiltonian.gradient(t)
