"""Class definition of a closed model."""

from collections.abc import Callable

import jax.numpy as jnp

from paraqeet.model.differentiable_hamiltonian import DifferentiableHamiltonian
from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.quantity import Array, Quantity


class ClosedSystem(EquationOfMotion):
    """Model of a closed physical system, defined by a Hamiltonian.

    Its dynamics is given by the Schrödinger equation.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Matrix representation of a Hamiltonian.

    """

    _get_value_method: Callable

    def __init__(self, hamiltonian: DifferentiableHamiltonian, ode_propagation: bool = False):
        super().__init__(hamiltonian)
        self.ode_propagation = ode_propagation

    @property
    def ode_propagation(self) -> bool:
        """Flag to set method of propagation to ODE.

        Returns
        -------
        bool
            Flag to use ODE propagation.
        """
        return self._ode_propagation

    @ode_propagation.setter
    def ode_propagation(self, ode_propagation: bool) -> None:
        self._ode_propagation = ode_propagation

        if ode_propagation:
            self._get_value_method = self.__get_ode_propagation_eom
        else:
            self._get_value_method = self.__get_eom

    def get_parameters(self) -> list[Quantity]:
        """Get a list of optimizable parameters.

        Returns
        -------
        List[Quantity]
            List of optimizable parameters of the system.

        """
        return self._hamiltonian.get_parameters()

    def __get_eom(self, times: Array) -> Array:
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
        return -1.0j * self._hamiltonian.get_value(times)

    def __get_ode_propagation_eom(self, times: Array) -> tuple[Array, Array]:
        """Get the matrix equations of motion for ODE solver.

        Here we return an empty array for the collapse operator.
        """
        return -1.0j * self._hamiltonian.get_value(times), jnp.empty((1,), dtype=jnp.complex128)

    def get_value(self, times: Array):
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
        return self._get_value_method(times)

    def get_gradient(self, times) -> Array:
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
        return -1.0j * self._hamiltonian.get_value_and_gradient(times)[1]
