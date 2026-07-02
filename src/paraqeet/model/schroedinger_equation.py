"""Class definition of a closed model."""

from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.quantity import Array


class SchroedingerEquation(EquationOfMotion):
    """Model of a closed physical system, defined by a Hamiltonian.

    Its dynamics is given by the Schrödinger equation.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Matrix representation of a Hamiltonian.

    """

    def get_value(self, times: Array) -> Array:
        """Get the matrix equations of motion.

        Computes the right hand side of the Schrödinger equation
        without multiplying the state.
        Used for unitary solvers.

        Parameters
        ----------
        times : Array
            Vector of time samples.

        Returns
        -------
        Array
            RHS with dimension [t, n, n]  with 't' as time
            and 'n' as Hilbert space dimension.

        """
        return -1.0j * self._hamiltonian_func(times)

    def get_value_and_gradient(self, times) -> tuple[Array, Array]:
        """Compute the gradient of getMatrix.

        Parameters
        ----------
        times : Array
            Vector of time samples.

        Returns
        -------
        Array
            Returns the gradient of getMatrix.

        """
        eom, eom_gradient = self._hamiltonian_and_gradient_func(times)
        return -1.0j * eom, -1.0j * eom_gradient
