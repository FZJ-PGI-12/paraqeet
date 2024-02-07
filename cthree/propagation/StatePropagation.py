from cthree.model.Model import Model

import numpy as np

from cthree.propagation.Propagation import Propagation


class StatePropagation(Propagation):
    """
    Abstract base class for all propagation implementation that need an initial state. This implements the
    setInitialState function.
    """

    _initialState: np.ndarray | None = None

    def __init__(self, model: Model):
        super().__init__(model)

    def setInitialState(self, state: np.ndarray):
        """
        Sets the initial state for the propagation. Subclasses can access the state in the _initialState field.

        :param state:
        :return:
        """
        self._initialState = state
