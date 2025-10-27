"""The class definition of state transfer fidelity model."""

import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import grad, jit

from paraqeet.differentiable import Differentiable
from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.propagation.state_propagation import StatePropagation
from paraqeet.quantity import Array

jax.config.update("jax_enable_x64", True)


class StateTransferFidelity(NormalizableMeasurement, Differentiable):
    """Fidelity measure that compares overlap of the initial and final state.

    Parameters
    ----------
    propagation : StatePropagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    initial_state : Array
        Initial state.
    target_state : Array
        Target state.
    times : Array
        One-dimensional vector of timestamps.

    """

    _initial_state: Array
    _target_state: Array
    _propagation: StatePropagation

    def __init__(
        self,
        propagation: StatePropagation,
        initial_state: Array,
        target_state: Array,
    ):
        self._propagation = propagation
        self._initial_state = initial_state
        self._target_state = target_state
        if target_state.shape != initial_state.shape:
            warnings.warn(
                UserWarning(
                    f"Different shapes for target_state({target_state.shape})"
                    f"and initial_state({initial_state.shape}) detected."
                    " Use restrict_subsystems to project states to "
                    "the same shape before measuring."
                )
            )
        self._propagation.set_initial_state(self._initial_state)
        if self._propagation.is_open:
            self._overlap = self._overlap_dm
        else:
            self._overlap = self._overlap_vec

    @staticmethod
    def _fid(overlap: Array) -> float:
        return float(jnp.abs(jnp.average(overlap)) ** 2)

    @staticmethod
    def _overlap_vec(target_state, final_state):
        return jnp.vdot(target_state, final_state)

    @staticmethod
    def _overlap_dm(target_state, final_state):
        return jnp.linalg.trace(jnp.matmul(target_state, final_state))

    # TODO: Check the implementation method measure
    def measure(self, times: Array) -> Array | float:
        """Return measurement in the range [0, 1]."""
        return self.calculate_normalized_scalar(times=times)

    def calculate_normalized_scalar(self, times: Array) -> float:
        """Measure overlap between initial and target state.
        
        Parameters
        ----------
        times : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Overlap between initial and target state in a JAX ArrayLike format.

        """
        states = self._propagation.propagate(time=times)
        final_state = states[-1]
        f = self._overlap(self._target_state, final_state)
        return self._fid(f)

    # TODO: adjust methods calling value_and_gradient
    def value_and_gradient(self, times: Array) -> tuple[Array, Array] | tuple[float, Array]:
        """Compute function value and corresponding gradient.

        Returns
        -------
        Tuple[Array, Array]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        states, dg_dp_list = self._propagation.gradient(time=times)
        final_state = states[-1]
        df_dp_list = []
        f = self._overlap(self._target_state, final_state)
        for dg_dp in dg_dp_list[-1]:
            g = self._overlap(self._target_state, dg_dp)
            df_dp_list.append(jnp.real(f.conj() * g + f * g.conj()))  # chain rule for abs^2
        return self._fid(f), jnp.array(df_dp_list)  # shape scalar, (n_parameters,)


class StateTransferFidelityAD(StateTransferFidelity):
    """Fidelity measure that compares overlap of the initial and final state.

    Parameters
    ----------
    propagation : Propagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    initial_state : Array
        Initial state.
    target_state : Array
        Target state.
    times : Array
        One-dimensional vector of timestamps.

    """

    __gradient_function: Callable | None

    def __init__(
        self,
        propagation: StatePropagation,
        initial_state: Array,
        target_state: Array,
    ):
        super().__init__(propagation, initial_state, target_state)
        self.__gradient_function = None

    def value_and_gradient(self, times: Array) -> tuple[Array, Array] | tuple[float, Array]:
        """Measure with gradient.

        Overwrite inherited `measureWithGradient` to calculate
        gradients using AD.

        Returns
        -------
        Tuple[float, Array]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        if self.__gradient_function is None:
            self.__gradient_function = jit(grad(self._fid, argnums=0))

        states, dg_dp_list = self._propagation.gradient(time=times)
        final_state = states[-1]
        df_dp_list = []
        f = self._overlap(self._target_state, final_state)
        for dg_dp in dg_dp_list[-1]:
            g = self._overlap(self._target_state, dg_dp)
            dfdp = self.__gradient_function(f) * g
            df_dp_list.append(jnp.real(dfdp))
        return self._fid(f), jnp.array(df_dp_list)  # shape scalar, (n_parameters,)


class StateTransferFidelityGRAPE(StateTransferFidelity):
    """Fidelity measure that compares overlap of the initial and final state.

    For GRAPE the optimizable parameters are vector quantities.

    Parameters
    ----------
    propagation : StatePropagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    initial_state : Array
        Initial state.
    target_state : Array
        Target state.
    times : Array
        One-dimensional vector of timestamps.

    """

    _propagation: StatePropagation

    def value_and_gradient(self, times: Array) -> tuple[Array, Array] | tuple[float, Array]:
        """Compute function value and corresponding gradient.

        Returns
        -------
        Tuple[Array, Array]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        states, grads = self._propagation.gradient(time=times)
        final_state = states[-1]
        f = self._overlap(self._target_state, final_state)
        if self._propagation.is_open:
            grads = jnp.real(jnp.linalg.trace(grads)).flatten()
        else:
            grads = 0.5 * jnp.real(f.conj() * grads + grads.conj() * f).flatten()
        return self._fid(f), grads  # shape scalar, (n_parameters,)
