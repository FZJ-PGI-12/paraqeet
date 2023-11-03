import numpy as np
import scipy.linalg as sclin

from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation
from cthree.Exceptions import IncompatibleLayersException


class MixedStateTransferFidelity(Measurement):
    """
    Fidelity measure that compares the overlap of the initial and final state of density matrices.

    Note: this implementation is still very inaccurate
    """
    __targetState: np.ndarray
    __targetStateSqrt: np.ndarray | None
    __propagation: Propagation
    __times: np.ndarray

    def __init__(self, propagation: Propagation, targetState: np.ndarray, times: np.ndarray):
        super().__init__()
        self.__propagation = propagation
        self.__targetState = targetState
        self.__times = times

        # store the sqrt of the density matrix to simplify the measurement
        self.__targetStateSqrt = sclin.sqrtm(self.__targetState)

    def measure(self) -> float:
        state = self.__propagation.propagate(self.__times)[-1]
        if state.shape != self.__targetState.shape:
            raise IncompatibleLayersException(
                f"Need a state vector of size {self.__targetState.shape} for the state transfer fidelity, but got shape {state.shape}")

        # density matrix
        product = self.__targetStateSqrt @ state @ self.__targetStateSqrt
        return np.abs(np.trace(sclin.sqrtm(product))) ** 2
