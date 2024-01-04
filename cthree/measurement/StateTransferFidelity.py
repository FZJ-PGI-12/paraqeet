from typing import List, Callable
import numpy as np
from cthree.Quantity import Quantity

from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation
from cthree.Exceptions import IncompatibleLayersException

import jax
import jax.numpy as jnp
from jax import grad
from functools import partial

jax.config.update("jax_enable_x64", True)


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

    def _computeMeasure(self, overlap):
        return np.abs(overlap) ** 2

    def measure(self) -> np.ndarray:
        states = self.__propagation.propagate(time=self._times)
        final_state = states[-1]
        overlap = np.vdot(self.__targetState, final_state)
        return self._computeMeasure(overlap)

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


class StateTransferFidelityAD(StateTransferFidelity):
    __gradientFunction: Callable

    def __init__(
        self,
        propagation: Propagation,
        initialState: np.ndarray,
        targetState: np.ndarray,
        times: np.ndarray,
    ):
        super().__init__(propagation, initialState, targetState, times)
        self.__propagation = propagation
        self.__initialState = initialState
        self.__targetState = targetState
        if targetState.shape != initialState.shape:
            raise IncompatibleLayersException(
                f"state vector of shape {self.__initialState.shape} needed for unitary fidelity"
            )
        self.__propagation.setInitialState(self.__initialState)
        self.__gradientFunction = None

    @partial(jax.jit, static_argnums=(0,))
    def _computeMeasure(self, overlap):
        """
        Overwrite inherited `_computeMeasure` function to make it JAX compatible.
        """
        return jnp.abs(overlap) ** 2

    def measureGradient(self) -> np.ndarray:
        """
        Overwrite inherited `measureGradient` to calculate gradients using AD.
        """

        if self.__gradientFunction is None:
            self.__gradientFunction = grad(self._computeMeasure, argnums=0)

        target_state = self.__targetState
        states = self.__propagation.propagate(time=self._times)
        final_state = states[-1]
        f = jnp.vdot(target_state, final_state)
        dg_dp_list = self.__propagation.gradient(time=self._times)
        dF_dp = []
        for dg_dp in dg_dp_list[-1]:
            g = jnp.vdot(target_state, dg_dp)
            dfdp = self.__gradientFunction(f) * g
            dF_dp.append(dfdp)
        return np.array(dF_dp)
