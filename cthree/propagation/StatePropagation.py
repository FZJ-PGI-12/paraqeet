"""Class definition of the State propagation model."""

from cthree.model.EquationOfMotion import EquationOfMotion

import numpy as np

from cthree.propagation.Propagation import Propagation


class StatePropagation(Propagation):
    """Propagation implementation that need an initial state.

    This implements the setInitialState function.

    Parameters
    ----------
    model : cthree.model.Model
        Represents the equation of motion for a given Hamiltonian.

    """

    _initialState: np.ndarray | None = None

    def __init__(self, model: EquationOfMotion):
        super().__init__(model)

    def setInitialState(self, state: np.ndarray):
        """Set the initial state for the propagation.

        Subclasses can access the state in the _initialState field.

        Parameters
        ----------
        state : numpy.ndarray
            Parameter value to be set as the initial state for the propagation.

        """
        self._initialState = state
