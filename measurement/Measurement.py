import numpy as np

from propagation.Propagation import Propagation


class Measurement:
    __propagation: Propagation

    def __init__(self, propagation: Propagation):
        self.__propagation = propagation

    def measure(self) -> float:
        pass

    def _getPropagator(self) -> np.array:
        return self.__propagation.propagate()
