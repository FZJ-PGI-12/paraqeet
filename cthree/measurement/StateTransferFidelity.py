import numpy as np

from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation
from cthree.Exceptions import IncompatibleLayersException


class StateTransferFidelity(Measurement):
    """
    Fidelity measure that compares the overlap of the initial and final state.
    """
    __initialState: np.ndarray
    __propagation: Propagation

    def __init__(self, propagation: Propagation, initialState: np.ndarray):
        super().__init__()
        self.__propagation = propagation
        self.__initialState = initialState

    def measure(self) -> float:
        state = self.__propagation.propagate()
        if state.shape != self.__initialState.shape:
            raise IncompatibleLayersException(
                f"state vector of size {len(self.__initialState)} needed for unitary fidelity")
        return 1.0 - np.vdot(self.__initialState, state)
