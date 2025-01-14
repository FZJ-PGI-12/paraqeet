"""Class definition of the Propagation model."""

from abc import abstractmethod

from cthree.Optimisable import Optimisable
from cthree.model.EquationOfMotion import EquationOfMotion

import numpy as np


class Propagation(Optimisable):
    """Abstract base class for any implementation of the equations of motion.

    The right-hand side of the equation is provided by the underlying model.

    Parameters
    ----------
    model : cthree.model.Model
        Represents the equation of motion for a given Hamiltonian.

    """

    _model: EquationOfMotion

    def __init__(self, model: EquationOfMotion):
        self._model = model

    def setInitialState(self, state: np.ndarray):
        """Set the initial state for the propagation.

        Propagation implementations that do not need the state should not
        implement this function.

        Parameters
        ----------
        state : numpy.ndarray
            Parameter value to be set as the initial state for the propagation.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    @abstractmethod
    def propagate(self, time: np.ndarray) -> np.ndarray:
        """Return the solution of the equations of motion.

        The first dimension of the result will always be the time.
        Like in the model, the format of the other dimensions depends on the
        implementation and could for example be a propagated state vector or
        a propagator in matrix form.

        Parameters
        ----------
        time : numpy.ndarray
            Any one-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Returns the solution of the equations of motion.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    def gradient(self, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute this part of the chain rule for a gradient trace.

        Computes the result of the propagation wrt model.
        The returned tuple contains the time-evolved state as well as the
        gradient. The time-dependent state is returned in the same shape
        as from the propagate method. In the gradient. the second dimension
        is the parameter index, i.e. result[i] will be the gradient at time t_i.

        Parameters
        ----------
        time : numpy.ndarray
            Any one-dimensional vector of timestamps.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            Computes part of the chain rune for a gradient trace.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()
