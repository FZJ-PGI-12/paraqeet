from abc import abstractmethod
from typing import List, Tuple

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
        Returns the solution of the equations of motion. The first dimension of the result will always be the time.
        Like in the model, the format of the other dimensions depends on the implementation and could for example be a
        propagated state vector or a propagator in matrix form.

        :param time: any one-dimensional vector of timestamps
        """
        raise NotImplementedError()

    def gradient(self, time: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Computes this part of the chain rule for a gradient trace. i.e. result of the propagation wrt model.
        """
        raise NotImplementedError()
