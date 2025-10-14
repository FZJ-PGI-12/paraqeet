"""Class definition of the Propagation model."""

from abc import ABC, abstractmethod

from paraqeet.quantity import Array

from paraqeet.model.equation_of_motion import EquationOfMotion


class Propagation(ABC):
    """Abstract base class for any implementation of the equations of motion.

    The right-hand side of the equation is provided by the underlying model.

    Parameters
    ----------
    model: Model
        Represents the equation of motion for a given Hamiltonian.

    """

    _model: EquationOfMotion | None

    def __init__(self, model: EquationOfMotion | None):
        self._model = model

    @abstractmethod
    def set_initial_state(self, state: Array):
        """Set the initial state for the propagation.

        Propagation implementations that do not need the state should not
        implement this function.

        Parameters
        ----------
        state: Array
            Parameter value to be set as the initial state for the propagation.

        """
        pass

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
        pass

    @abstractmethod
    def gradient(self, time: Array) -> tuple[Array, Array]:
        """Compute this part of the chain rule for a gradient trace.

        Computes the result of the propagation wrt model.
        The returned tuple contains the time-evolved state as well as the
        gradient. The time-dependent state is returned in the same shape
        as from the propagate method. In the gradient. the second dimension
        is the parameter index, i.e. result[i] will be the gradient at time t_i.

        Parameters
        ----------
        time: Array
            Any one-dimensional vector of timestamps.

        Returns
        -------
        tuple[Array Array]
            Computes part of the chain rune for a gradient trace.


        """
        pass
