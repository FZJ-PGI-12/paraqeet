from abc import abstractmethod
from typing import List

from cthree.Optimisable import Optimisable
from cthree.model.Model import Model

import numpy as np


class Propagation(Optimisable):
    """
    Abstract base class for any implementation that can solve the equation of motion. The right-hand side of the
    equation is provided by the underlying model.
    """

    _model: Model

    def __init__(self, model: Model):
        self._model = model

    def setInitialState(self, state: np.ndarray):
        """
        Sets the initial state for the propagation. Propagation implementations that do not need the state should not
        implement this function.

        :param state:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def propagate(self, time: np.ndarray) -> np.ndarray:
        """
        Returns the solution of the equations of motion. Like in the model, the format of the result depends on the
        implementation and could for example be a propagated state vector or a propagator in matrix form.

        :param time: any one-dimensional vector of timestamps
        """
        raise NotImplementedError()
