"""Class definition of a closed model."""

from collections.abc import Callable

import jax.numpy as jnp

from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.model.solvable import Solvable
from paraqeet.quantity import Array, Quantity


class SchroedingerEquation(EquationOfMotion):
    """Model of a closed physical system, defined by a Hamiltonian.

    Its dynamics is given by the Schrödinger equation.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Matrix representation of a Hamiltonian.

    """

    _get_value_method: Callable

    def __init__(self, solvable: Solvable, ode_propagation: bool = False):
        super().__init__(solvable)
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
            self._get_value_method = self._get_ode_propagation_eom
        else:
            self._get_value_method = self._get_eom

    def _get_eom(self, times: Array) -> Array:
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
        return -1.0j * self._solvable.get_value(times)

    def _get_ode_propagation_eom(self, times: Array) -> tuple[Array, Array]:
        """Get the matrix equations of motion for ODE solver.

        Here we return an empty array for the collapse operator.
        """
        return -1.0j * self._solvable.get_value(times), jnp.empty((1,), dtype=jnp.complex128)

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
        eom, eom_gradient = self._solvable.get_value_and_gradient(times)
        return -1.0j * eom, -1.0j * eom_gradient
