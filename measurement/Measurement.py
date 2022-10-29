import numpy as np

from propagation.Propagation import Propagation


class Measurement:
    __propagation: Propagation

    def measure(self) -> float:
        pass

    def _getPropagator(self) -> np.array:
        return self.__propagation.propagate()