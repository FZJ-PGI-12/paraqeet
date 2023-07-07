import numpy as np

from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation
from cthree.Exceptions import IncompatibleLayersException
from cthree.QuantumState import QuantumState


class StateTransferFidelity(Measurement):
    """
    Fidelity measure that compares the overlap of the initial and final state.
    """

    __initialState: QuantumState
    __targetState: QuantumState
    __propagation: Propagation

    def __init__(
        self,
        propagation: Propagation,
        initialState: QuantumState,
        targetState: QuantumState,
    ):
        super().__init__()
        self.__propagation = propagation
        self.__initialState = initialState
        self.__targetState = targetState


    def measure(self) -> float:
        state = self.__propagation.propagate(
            init=self.__initialState, time=self.__targetState.getTime()
        )
        """
        TODO: Avoid unwrapping the QuantumState class by passing properties directly.
        """
        if state.shape != self.__initialState.shape:
            raise IncompatibleLayersException(
                f"state vector of shape {self.__initialState.shape} needed for unitary fidelity"
            )
        return 1 - np.abs(np.vdot(self.__targetState.getVector(), state)) ** 2
