from typing import List, Tuple, Callable
import numpy as np
from cthree.Quantity import Quantity

from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation
from cthree.Exceptions import IncompatibleLayersException

import jax
import jax.numpy as jnp
from jax import grad, jit

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

    @staticmethod
    def _fid(overlap):
        return np.abs(overlap) ** 2

    def measure(self) -> np.ndarray:
        states = self.__propagation.propagate(time=self._times)
        final_state = states[-1]
        f = np.vdot(self.__targetState, final_state)
        return self._fid(f)

    def measureWithGradient(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute function value and corresponding gradient.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of function value and gradient of shape (n_parameters,)
        """
        states, dg_dp_list = self.__propagation.gradient(time=self._times)
        final_state = states[-1]
        dF_dp = []
        f = np.vdot(self.__targetState, final_state)
        for dg_dp in dg_dp_list[-1]:
            g = np.vdot(self.__targetState, dg_dp)
            dF_dp.append(f.conj() * g + f * g.conj())  # chain rule for abs^2
        return self._fid(f), np.array(dF_dp)  # shape scalar, (n_parameters,)

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

    @staticmethod
    @jit
    def _fid(overlap):
        """
        Overwrite inherited `_fid` function to make it JAX compatible.
        """
        return jnp.abs(overlap) ** 2

    def measureWithGradient(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Overwrite inherited `measureWithGradient` to calculate gradients using AD.
        """

        if self.__gradientFunction is None:
            self.__gradientFunction = jit(grad(self._fid, argnums=0))

        states, dg_dp_list = self.__propagation.gradient(time=self._times)
        final_state = states[-1]
        dF_dp = []
        f = jnp.vdot(self.__targetState, final_state)
        for dg_dp in dg_dp_list[-1]:
            g = jnp.vdot(self.__targetState, dg_dp)
            dfdp = self.__gradientFunction(f) * g
            dF_dp.append(dfdp)
        return self._fid(f), np.array(dF_dp)  # shape scalar, (n_parameters,)
