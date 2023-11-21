from typing import List

import numpy as np

from cthree.propagation.Propagation import Propagation


class IdentityPropagation(Propagation):
    """
    Mock propagation implementation that returns the initial state as the target state.
    """

    __state: np.ndarray

    def __init__(self):
        super().__init__(None)

    def setInitialState(self, state: np.ndarray):
        self.__state = state

    def propagate(self, time: np.ndarray) -> List[np.ndarray]:
        return [self.__state] * len(time)
