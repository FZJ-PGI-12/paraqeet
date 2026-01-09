"""Class definition of the optimizable model."""

from abc import abstractmethod

from paraqeet.model.differentiable_hamiltonian import DifferentiableHamiltonian
from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Array


class EquationOfMotion(Optimizable):
    """Represents the equation of motion for a given Hamiltonian.

    Implementations can for example be the Schrödinger equation for a
    closed system, Lindbladian for an open system, or Hamilton's equations
    for a classical system.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Matrix representation of a Hamiltonian.

    """

    _hamiltonian: DifferentiableHamiltonian

    def __init__(self, hamiltonian: DifferentiableHamiltonian):
        self._hamiltonian = hamiltonian

    def get_right_hand_side(self, time: Array, state: Array) -> Array:
        """Return the right-hand side of the equations of motion.

        The format depends on the implementation and could for example
        be a state vector or a matrix. Default implementation assumes a
        homogeneous ODE with matrix operator given by self.getMatrix().

        Parameters
        ----------
        time: Array
            Any one-dimensional vector of timestamps.

        Returns
        -------
        Array
            The right-hand side of the equation of motion at each time stamp.

        """
        return self.get_value(time) @ state

    @abstractmethod
    def get_value(self, time: Array) -> Array:
        """Abstract method to get the prefactor matrix.

        Parameters
        ----------
        time: Array
            Any one-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns matrix equations of motion.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        pass

    @abstractmethod
    def get_gradient(self, t: Array) -> Array:
        """Implement the gradient of either getEquationOfMotion or getMatrixEOM.

        Parameters
        ----------
        t: Array
            Any one-dimensional vector of timestamps.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        pass
