"""Class definition of the State propagation model."""

from abc import ABC

from cthree.quantity import Array

from cthree.model.equation_of_motion import EquationOfMotion
from cthree.propagation.propagation import Propagation


class StatePropagation(Propagation, ABC):
    """Propagation implementation that need an initial state.

    This implements the set_initial_state function.

    Parameters
    ----------
    model : cthree.model.Model
        Represents the equation of motion for a given Hamiltonian.

    """

    _initial_state: Array | None = None

    def __init__(self, model: EquationOfMotion):
        super().__init__(model)

    def set_initial_state(self, state: Array):
        """Set the initial state for the propagation.

        Subclasses can access the state in the _initial_sate field.

        Parameters
        ----------
        state : numpy.ndarray
            Parameter value to be set as the initial state for the propagation.

        """
        self._initial_state = state
