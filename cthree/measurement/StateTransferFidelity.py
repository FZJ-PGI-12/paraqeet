from typing import List, Tuple, Callable
from cthree.Quantity import Quantity

from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation
from cthree.Exceptions import IncompatibleLayersException

import jax
import jax.numpy as jnp
from jax import Array, grad, jit
from jax.typing import ArrayLike

import warnings

jax.config.update("jax_enable_x64", True)


class StateTransferFidelity(Measurement):
    """
    Fidelity measure that compares the overlap of the initial and final state.
    """

    __initialState: ArrayLike
    __targetState: ArrayLike
    _times: ArrayLike
    __propagation: Propagation

    def __init__(
        self,
        propagation: Propagation,
        initialState: ArrayLike,
        targetState: ArrayLike,
        times: ArrayLike,
    ):
        super().__init__(times=times)
        self.__propagation = propagation
        self.__initialState = initialState
        self.__targetState = targetState
        if targetState.shape != initialState.shape:
            warnings.warn(
                f"""
                Different shapes for targetState({targetState.shape}) and initialState({initialState.shape}) detected.
                Use restrictSubsystems to project states to the same shape before measuring.
                """
            )
        self.__propagation.setInitialState(self.__initialState)

    @staticmethod
    def _fid(overlap):
        return jnp.abs(overlap) ** 2

    def measure(self) -> ArrayLike:
        states = self.__propagation.propagate(time=self._times)
        states = self._preprocess(states)
        final_state = states[-1]
        f = jnp.vdot(self.__targetState, final_state)
        return self._fid(f)

    def measureWithGradient(self) -> Tuple[Array, Array]:
        """
        Compute function value and corresponding gradient.

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            Tuple of function value and gradient of shape (n_parameters,)
        """
        states, dg_dp_list = self.__propagation.gradient(time=self._times)
        states = self._preprocess(states)
        final_state = states[-1]
        dF_dp = []
        f = jnp.vdot(self.__targetState, final_state)
        for dg_dp in dg_dp_list[-1]:
            g = jnp.vdot(self.__targetState, dg_dp)
            dF_dp.append(jnp.real(f.conj() * g + f * g.conj()))  # chain rule for abs^2
        return self._fid(f), jnp.array(dF_dp)  # shape scalar, (n_parameters,)

    def getParameters(self) -> List[Quantity]:
        return []


class StateTransferFidelityAD(StateTransferFidelity):
    __gradientFunction: Callable | None

    def __init__(
        self,
        propagation: Propagation,
        initialState: ArrayLike,
        targetState: ArrayLike,
        times: ArrayLike,
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

    def measureWithGradient(self) -> Tuple[Array, Array]:
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
            dF_dp.append(jnp.real(dfdp))
        return self._fid(f), jnp.array(dF_dp)  # shape scalar, (n_parameters,)
