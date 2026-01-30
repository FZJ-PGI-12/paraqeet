"""Class definition of the Propagation model."""

from abc import ABC, abstractmethod

from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.model.open_system import OpenSystem
from paraqeet.quantity import Array


class Propagation(ABC):
    """Abstract base class for any implementation of the equations of motion.

    The right-hand side of the equation is provided by the underlying model.

    Parameters
    ----------
    model: Model
        Represents the equation of motion for a given Hamiltonian.

    """

    _model: EquationOfMotion | None
    _initial_state: Array | None = None
    _is_open: bool = False

    def __init__(self, model: EquationOfMotion | None):
        self._model = model
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

    # TODO: Keep or remove get_parameters?
    # @staticmethod
    def get_parameters(self):
        """Per default, propagation methods have no parameters to optimize."""
        return []

    def set_initial_state(self, state: Array):
        """Set the initial state for the propagation. (Default implementation)

        Propagation implementations that do not need the state should not
        implement this function.

        Subclasses can access the state in the _initial_sate field.

        Parameters
        ----------
        state: Array
            Parameter value to be set as the initial state for the propagation.

        """
        self._initial_state = state

    @abstractmethod
    def propagate(self, time: Array) -> Array:
        """Return the solution of the equations of motion.

        The first dimension of the result will always be the time.
        Like in the model, the format of the other dimensions depends on the
        implementation and could for example be a propagated state vector or
        a propagator in matrix form.

        Parameters
        ----------
        time: Array
            Any one-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the solution of the equations of motion.


        """
        # TODO: Distinguish between internal time (class property), i.e. the time grid of the
        # method vs. time points (input parameter) desired by other classes, e.g. Measurements
        pass
