"""Class definition of the State propagation model."""

from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.model.open_system import OpenSystem
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array
from abc import abstractmethod


class StatePropagation(Propagation):
    """Propagation implementation that need an initial state.

    This implements the set_initial_state function.

    Parameters
    ----------
    model: Model
        Represents the equation of motion for a given Hamiltonian.

    """

    _initial_state: Array | None = None
    _is_open: bool = False

    def __init__(self, model: EquationOfMotion):
        super().__init__(model)
        if isinstance(model, OpenSystem):
            self.is_open = True

    @property
    def is_open(self) -> bool:
        """Return if the propagation is for open or closed system."""
        return self._is_open

    @is_open.setter
    def is_open(self, flag) -> None:
        """Set if the propagation is for open or closed system."""
        self._is_open = flag

    def set_initial_state(self, state: Array):
        """Set the initial state for the propagation.

        Subclasses can access the state in the _initial_sate field.

        Parameters
        ----------
        state: Array
            Parameter value to be set as the initial state for the propagation.

        """
        self._initial_state = state

    @abstractmethod
    def gradient(self, time: Array) -> tuple[Array, Array]:
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
