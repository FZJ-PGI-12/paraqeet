"""The class definition of state transfer fidelity model."""

from collections.abc import Callable
from cthree.Quantity import Quantity

from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation

import jax
import jax.numpy as jnp
from jax import Array, grad, jit
from jax.typing import ArrayLike

import warnings

jax.config.update("jax_enable_x64", True)


class StateTransferFidelity(Measurement):
    """Fidelity measure that compares overlap of the initial and final state.

    Parameters
    ----------
    propagation : cthree.measurement.Propagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    initialState : jax.typing.ArrayLike
        Initial state.
    targetState : jax.typing.ArrayLike
        Target state.
    times : jax.typing.ArrayLike
        One-dimensional vector of timestamps.

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
                UserWarning(
                    f"Different shapes for targetState({targetState.shape})"
                    f"and initialState({initialState.shape}) detected."
                    " Use restrictSubsystems to project states to "
                    "the same shape before measuring."
                )
            )
        self.__propagation.setInitialState(self.__initialState)

    @staticmethod
    def _fid(overlap):
        return jnp.abs(overlap) ** 2

    def measure(self) -> ArrayLike:
        """Measure overlap between initial and target state.

        Returns
        -------
        jax.typing.ArrayLike
            Overlap between initial and target state in a JAX ArrayLike format.

        """
        states = self.__propagation.propagate(time=self._times)
        states = self._preprocessVector(states)
        final_state = states[-1]
        f = jnp.vdot(self.__targetState, final_state)
        return self._fid(f)

    def measureWithGradient(self) -> tuple[Array, Array]:
        """Compute function value and corresponding gradient.

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        states, dg_dp_list = self.__propagation.gradient(time=self._times)
        states = self._preprocessVector(states)
        dg_dp_list = self._preprocessVector(dg_dp_list)
        final_state = states[-1]
        dF_dp = []
        f = jnp.vdot(self.__targetState, final_state)
        for dg_dp in dg_dp_list[-1]:
            g = jnp.vdot(self.__targetState, dg_dp)
            dF_dp.append(
                jnp.real(f.conj() * g + f * g.conj())
            )  # chain rule for abs^2
        return self._fid(f), jnp.array(dF_dp)  # shape scalar, (n_parameters,)

    def getParameters(self) -> list[Quantity]:
        """Get the parameters of the system.

        Returns
        -------
        list[cthree.Quantity]
            List of parameters of the system.
        """
        return []


class StateTransferFidelityAD(StateTransferFidelity):
    """Fidelity measure that compares overlap of the initial and final state.

    Parameters
    ----------
    propagation : cthree.propagation.Propagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    initialState : jax.typing.ArrayLike
        Initial state.
    targetState : jax.typing.ArrayLike
        Target state.
    times : jax.typing.ArrayLike
        One-dimensional vector of timestamps.

    """

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
            warnings.warn(
                UserWarning(
                    f"Different shapes for targetState({targetState.shape})"
                    f"and initialState({initialState.shape}) detected."
                    " Use restrictSubsystems to project states to the"
                    " same shape before measuring."
                )
            )
        self.__propagation.setInitialState(self.__initialState)
        self.__gradientFunction = None

    def measureWithGradient(self) -> tuple[Array, Array]:
        """Measure with gradient.

        Overwrite inherited `measureWithGradient` to calculate
        gradients using AD.

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        if self.__gradientFunction is None:
            self.__gradientFunction = jit(grad(self._fid, argnums=0))

        states, dg_dp_list = self.__propagation.gradient(time=self._times)
        states = self._preprocessVector(states)
        dg_dp_list = self._preprocessVector(dg_dp_list)
        final_state = states[-1]
        dF_dp = []
        f = jnp.vdot(self.__targetState, final_state)
        for dg_dp in dg_dp_list[-1]:
            g = jnp.vdot(self.__targetState, dg_dp)
            dfdp = self.__gradientFunction(f) * g
            dF_dp.append(jnp.real(dfdp))
        return self._fid(f), jnp.array(dF_dp)  # shape scalar, (n_parameters,)


class StateTransferFidelityGRAPE(StateTransferFidelity):
    """Fidelity measure that compares overlap of the initial and final state.

    For GRAPE the optimisable parameters are vector quantities.

    Parameters
    ----------
    propagation : cthree.measurement.Propagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    initialState : jax.typing.ArrayLike
        Initial state.
    targetState : jax.typing.ArrayLike
        Target state.
    times : jax.typing.ArrayLike
        One-dimensional vector of timestamps.

    """

    __initialState: jnp.ndarray
    __targetState: jnp.ndarray
    _times: jnp.ndarray
    __propagation: Propagation

    def __init__(
        self,
        propagation: Propagation,
        initialState: jnp.ndarray,
        targetState: jnp.ndarray,
        times: jnp.ndarray,
    ):
        super().__init__(propagation, initialState, targetState, times)
        self.__propagation = propagation
        self.__initialState = initialState
        self.__targetState = targetState
        if targetState.shape != initialState.shape:
            warnings.warn(
                UserWarning(
                    f"Different shapes for targetState({targetState.shape})"
                    f"and initialState({initialState.shape}) detected."
                    " Use restrictSubsystems to project states to "
                    "the same shape before measuring."
                )
            )
        self.__propagation.setInitialState(self.__initialState)

    def measureWithGradient(self) -> tuple[Array, Array]:
        """Compute function value and corresponding gradient.

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        states, grads = self.__propagation.gradient(time=self._times)
        final_state = states[-1]
        f = jnp.vdot(self.__targetState, final_state)
        grads = jnp.real(f.conj() * grads + grads.conj() * f).flatten()
        return self._fid(f), grads  # shape scalar, (n_parameters,)
