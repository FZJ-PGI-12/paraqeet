import numpy as np

from measurement.Measurement import Measurement


class UnitaryFidelity(Measurement):
    __gate: np.array

    def __construct(self, gate: np.array):
        self.__gate = gate

    def measure(self) -> float:
        U = self._getPropagator()
        return 1.0 - np.trace(np.conjugate(self.__gate.T) * U)
