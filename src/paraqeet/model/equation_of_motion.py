"""Class definition of the optimizable model."""

from abc import ABC, abstractmethod
from collections.abc import Callable

from paraqeet.quantity import Array


class EquationOfMotion(ABC):
    """Represents the equation of motion for a given Hamiltonian.

    Implementations can for example be the Schrödinger equation for a
    closed system, Lindbladian for an open system, or Hamilton's equations
    for a classical system.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Matrix representation of a Hamiltonian.

    """

    _hamiltonian_func: Callable[[Array], Array]
    _hamiltonian_and_gradient_func: Callable[[Array], tuple[Array, Array]]

    def __init__(
        self,
        hamiltonian_func: Callable[[Array], Array],
        hamiltonian_and_gradient_func: Callable[[Array], tuple[Array, Array]],
    ):
        self._hamiltonian_func = hamiltonian_func
        self._hamiltonian_and_gradient_func = hamiltonian_and_gradient_func

    @abstractmethod
    def get_value(self, times: Array) -> Array:
        """Abstract method to get the prefactor matrix.

        Parameters
        ----------
        times: Array
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

    # TODO: Since this method delegates the call to the Hamiltonian-instance, should
    # we rename it to get_habiltonian_value_and_gradient or similar? Otherwise it suggests
    # that it returns the gradient of the equation of motion itself.
    @abstractmethod
    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
        """Implement the gradient of either getEquationOfMotion or getMatrixEOM.

        Parameters
        ----------
        times: Array
            Any one-dimensional vector of timestamps.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        pass
