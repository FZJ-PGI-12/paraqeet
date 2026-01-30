"""Class definition of the State propagation model."""

from abc import abstractmethod
from paraqeet.differentiable import Differentiable
from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array


class DifferentiablePropagation(Propagation, Differentiable):
    """Propagation methods that provide a get_value_and_gradient method.

    Parameters
    ----------
    model: Model
        Represents the equation of motion for a given Hamiltonian.
    """

    _initial_state: Array | None = None
    _is_open: bool = False

    def __init__(self, model: EquationOfMotion):
        super().__init__(model)

    @abstractmethod
    def get_value_and_gradient(self, time: Array) -> tuple[Array, Array]:
        """Gradient method to be implemented

        Parameters
        ----------
        time: Array
            Array of timesteps.

        Returns
        -------
        tuple[Array, Array]
            First dimension is time, second dimension is the parameter.

        """
        pass
