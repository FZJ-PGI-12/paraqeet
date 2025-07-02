"""Class definition of the optimisable model."""

from abc import abstractmethod

from paraqeet.quantity import Array

from paraqeet.model.hamiltonian import Hamiltonian
from paraqeet.optimisable import Optimisable
from paraqeet.quantity import Quantity


class EquationOfMotion(Optimisable):
    """Represents the equation of motion for a given Hamiltonian.

    Implementations can for example be the Schrödinger equation for a
    closed system, Lindbladian for an open system, or Hamilton's equations
    for a classical system.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Matrix representation of a Hamiltonian.

    """

    _hamiltonian: Hamiltonian

    def __init__(self, hamiltonian: Hamiltonian):
        self._hamiltonian = hamiltonian

    @abstractmethod
    def get_parameters(self) -> list[Quantity]:
        """Abstract method to get parameters of the model.

        Returns
        -------
        List[Quantity]
            Returns the list of parameters as Quantities.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    def get_right_hand_side(self, time: Array, state: Array) -> Array:
        """Return the right-hand side of the equations of motion.

        The format depends on the implementation and could for example
        be a state vector or a matrix. Default implementation assumes a
        homogeneous ODE with matrix operator given by self.getMatrix().

        Parameters
        ----------
        time : Array
            Any one-dimensional vector of timestamps.

        Returns
        -------
        Array
            The right-hand side of the equation of motion at each time stamp.

        """
        return self.get_matrix(time) @ state

    @abstractmethod
    def get_matrix(self, time: Array) -> Array:
        """Abstract method to get the prefactor matrix.

        Parameters
        ----------
        time : Array
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
        raise NotImplementedError()

    @abstractmethod
    def gradient(self, t) -> Array:
        """Implement the gradient of either getEquationOfMotion or getMatrixEOM.

        Parameters
        ----------
        t
            Any one-dimensional vector of timestamps.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()
