from typing import List
import numpy as np
from cthree.Quantity import Quantity

from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation
from cthree.Exceptions import IncompatibleLayersException


class StateTransferFidelity(Measurement):
    """
    Fidelity measure that compares the overlap of the initial and final state.
    """

    __initialState: np.ndarray
    __targetState: np.ndarray
    _times: np.ndarray
    __propagation: Propagation

    def __init__(
        self,
        propagation: Propagation,
        initialState: np.ndarray,
        targetState: np.ndarray,
        times: np.ndarray,
    ):
        super().__init__(times=times)
        self.__propagation = propagation
        self.__initialState = initialState
        self.__targetState = targetState
        if targetState.shape != initialState.shape:
            raise IncompatibleLayersException(
                f"state vector of shape {self.__initialState.shape} needed for unitary fidelity"
            )
        self.__propagation.setInitialState(self.__initialState)

    def measure(self) -> np.ndarray:
        states = self.__propagation.propagate(time=self._times)
        final_state = states[-1]
        return np.abs(np.vdot(self.__targetState, final_state)) ** 2

    def measureGradient(self) -> np.ndarray:
        states = self.__propagation.propagate(time=self._times)
        final_state = states[-1]
        dg_dp_list = self.__propagation.gradient(time=self._times)
        dF_dp = []
        f = np.vdot(self.__targetState, final_state)
        for dg_dp in dg_dp_list[-1]:
            g = np.vdot(self.__targetState, dg_dp)
            dF_dp.append(f.conj() * g + f * g.conj())
        return np.array(dF_dp)

    def getParameters(self) -> List[Quantity]:
        return []
