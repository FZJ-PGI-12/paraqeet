"""Class definition of a closed model."""

from typing import override

from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.quantity import Array


class SchroedingerEquation(EquationOfMotion):
    """Model of a closed physical system, defined by a Hamiltonian.

    Its dynamics is given by the Schrödinger equation.

    """

    @override
    def get_value(self, times: Array) -> Array:
        """Computes the right hand side of the Schrödinger equation
        without multiplying the state.

        Parameters
        ----------
        times: Array
            Array of times

        Returns
        -------
        Array
            RHS with dimension [t, n, n]  with 't' as time
            and 'n' as Hilbert space dimension.

        """
        return -1.0j * self._hamiltonian_func(times)

    @override
    def get_gradient(self, times: Array) -> Array:
        """Compute the gradient of right hand side of the Schrödinger equation
        without multiplying the state.


        Parameters
        ----------
        times: Array
            Array of times.

        Returns
        -------
        Array
            Returns the gradient of getMatrix.

        """
        eom_gradient = self._hamiltonian_gradient_func(times)
        return -1.0j * eom_gradient
