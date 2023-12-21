from typing import List, Callable

from cthree.Quantity import Quantity
from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation
from cthree.Exceptions import IncompatibleLayersException

from functools import partial

import jax
import jax.numpy as np
from jax import grad

jax.config.update("jax_enable_x64", True)


class StateTransferFidelityAD(Measurement):
    """
    Fidelity measure that compares the overlap of the initial and final state.
    """

    __initialState: np.ndarray
    __targetState: np.ndarray
    __times: np.ndarray
    __propagation: Propagation
    __gradientFunction: Callable

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
        self.__gradientFunction = grad(self.__computeMeasure, argnums=0)

    def measure(self) -> float:
        states = self.__propagation.propagate(time=self._times)
        final_state = np.expand_dims(states[-1], axis=1)
        target_state = self.__targetState
        overlap = self.__computeoverlap(final_state, target_state)
        return self.__computeMeasure(overlap)

    @partial(jax.jit, static_argnums=(0,))
    def __computeoverlap(self, target_state, final_state):
        return np.vdot(target_state, final_state)

    @partial(jax.jit, static_argnums=(0,))
    def __computeMeasure(self, overlap):
        return np.abs(overlap) ** 2

    def measureGradient(self) -> np.ndarray:
        target_state = self.__targetState
        states = self.__propagation.propagate(time=self._times)
        final_state = states[-1]
        f = self.__computeoverlap(target_state, final_state)
        dg_dp_list = self.__propagation.gradient(time=self._times)
        dF_dp = []
        for dg_dp in dg_dp_list[-1]:
            g = self.__computeoverlap(target_state, dg_dp)
            dfdp = self.__gradientFunction(f) * g
            dF_dp.append(dfdp)
        return np.array(dF_dp)

    def getParameters(self) -> List[Quantity]:
        return []
