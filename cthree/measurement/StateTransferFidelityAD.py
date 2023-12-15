from typing import List
import jax.numpy as np
from jax import grad
from cthree.Quantity import Quantity

from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation
from cthree.Exceptions import IncompatibleLayersException


class StateTransferFidelityAD(Measurement):
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
        super().__init__(times=times)
        self.__propagation = propagation
        self.__initialState = initialState
        self.__targetState = targetState
        if targetState.shape != initialState.shape:
            raise IncompatibleLayersException(
                f"state vector of shape {self.__initialState.shape} needed for unitary fidelity"
            )
        self.__propagation.setInitialState(self.__initialState)

    def measure(self) -> float:
        states = self.__propagation.propagate(time=self._times)
        final_state = np.expand_dims(states[-1], axis=1)
        target_state = self.__targetState
        overlap = self.__computeoverlap(final_state, target_state)
        return self.__computeMeasure(overlap)

    def __computeoverlap(self, final_state, target_state):
        return np.vdot(target_state, final_state)

    def __computeMeasure(self, overlap):
        return np.abs(overlap) ** 2

    def measureGradient(self) -> np.ndarray:
        target_state = self.__targetState
        states = self.__propagation.propagate(time=self._times)
        final_state = states[-1]
        overlap = self.__computeoverlap(final_state, target_state)
        dg_dc_list = self.__propagation.gradient(time=self._times)
        dF_dp = []
        for dg_dc in dg_dc_list:
            final_state_grad = dg_dc[-1]
            doverlap = self.__computeoverlap(final_state_grad, target_state)
            dfdc = grad(self.__computeMeasure, argnums=0)(overlap) * doverlap
            dF_dp.append(dfdc)
        return np.array(dF_dp)

    def getParameters(self) -> List[Quantity]:
        return []
