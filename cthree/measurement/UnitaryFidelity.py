import numpy as np

from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation
from cthree.Exceptions import IncompatibleLayersException


class UnitaryFidelity(Measurement):
    """
    Fidelity measure that compares the propagator with a desired gate.
    """
    __gate: np.ndarray
    __propagation: Propagation

    def __init__(self, propagation: Propagation, gate: np.ndarray):
        super().__init__()
        self.__propagation = propagation
        self.__gate = gate

    def measure(self) -> float:
        U = self.__propagation.propagate()
        if U.shape != self.__gate.shape:
            raise IncompatibleLayersException("propagator needed for unitary fidelity")
        return 1.0 - np.trace(np.conjugate(self.__gate.T) * U)
