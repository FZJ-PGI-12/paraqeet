import numpy as np

from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation
from cthree.Exceptions import IncompatibleLayersException


class StateTransferFidelity(Measurement):
    """
    Fidelity measure that compares the overlap of the initial and final state.
    """

    __initialState: np.ndarray
    __targetState: np.ndarray
    __times: np.ndarray
    __propagation: Propagation

    def __init__(
        self,
        propagation: Propagation,
        initialState: np.ndarray,
        targetState: np.ndarray,
        times: np.ndarray,
    ):
        super().__init__()
        self.__propagation = propagation
        self.__initialState = initialState
        self.__targetState = targetState
        self.__times = times

    def measure(self) -> float:
        states = self.__propagation.propagate(
            init=self.__initialState, time=self.__times
        )
        final_state = states[-1]
        if final_state.shape != self.__initialState.shape:
            raise IncompatibleLayersException(
                f"state vector of shape {self.__initialState.shape} needed for unitary fidelity"
            )
        return np.abs(np.vdot(self.__targetState, final_state)) ** 2
