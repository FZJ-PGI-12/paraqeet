import numpy as np

from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation


class UnitaryFidelity(Measurement):
    __gate: np.array

    def __init__(self, propagation: Propagation, gate: np.array):
        super().__init__(propagation)
        self.__gate = gate

    def measure(self) -> float:
        U = self._getPropagator()
        return 1.0 - np.trace(np.conjugate(self.__gate.T) * U)
