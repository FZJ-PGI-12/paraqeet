"""The class definition of state transfer fidelity model."""

import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import grad, jit

from paraqeet.differentiable import Differentiable
from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.propagation.propagation import DifferentiablePropagation
from paraqeet.quantity import Array

jax.config.update("jax_enable_x64", True)


class StateTransferFidelity(NormalizableMeasurement, Differentiable):
    """Fidelity measure that compares overlap of the initial and final state.

    Parameters
    ----------
    propagation : DifferentiablePropagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    target_state : Array
        Target state.
    times : Array
        One-dimensional vector of timestamps.

    """

    _target_state: Array
    _overlap: Callable
    _fid_grad_func: Callable
    _propagation_func: Callable

    def __init__(
        self,
        propagation_func: Callable,
        propagation_and_gradient_func: Callable,
        target_state: Array,
        overlap: Callable,
    ):
        self._propagation_func = propagation_func
        self._propagation_and_gradient_func = propagation_and_gradient_func
        self._target_state = target_state
        self._overlap = overlap

    @staticmethod
    def _fid(overlap: Array) -> float:
        return float(jnp.abs(jnp.average(overlap)) ** 2)

    def measure(self, times: Array) -> Array | float:
        """Return measurement in the range [0, 1]."""
        return self.calculate_normalized_scalar(times=times)

    def calculate_normalized_scalar(self, times: Array) -> float:
        """Measure overlap between initial and target state. To be used with an optimizer.

        Parameters
        ----------
        times : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        float
            Overlap between initial and target state in a bare float.

        """
        states = self._propagation_func(times)
        final_state = states[-1]
        f = self._overlap(self._target_state, final_state)
        return StateTransferFidelity._fid(f)

    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array] | tuple[float, Array]:
        """Compute function value and corresponding gradient.

        Returns
        -------
        Tuple[Array, Array]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        states, dg_dp_list = self._propagation_and_gradient_func(times)
        final_state = states[-1]
        df_dp_list = []
        f = self._overlap(self._target_state, final_state)
        for dg_dp in dg_dp_list[-1]:
            g = self._overlap(self._target_state, dg_dp)
            df_dp_list.append(jnp.real(f.conj() * g + f * g.conj()))  # chain rule for abs^2
        return StateTransferFidelity._fid(f), jnp.array(df_dp_list)  # shape scalar, (n_parameters,)


# TODO: Integrate AD version so you don't have to provide a gradient for _overlap
class StateTransferFidelityAD(StateTransferFidelity):
    """Fidelity measure that compares overlap of the initial and final state.

    Parameters
    ----------
    propagation : Propagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    target_state : Array
        Target state.
    times : Array
        One-dimensional vector of timestamps.

    """

    _gradient_function: Callable | None = None

    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array] | tuple[float, Array]:
        """Measure with gradient.

        Overwrite inherited `measure_with_gradient` to calculate
        gradients using AD.

        Returns
        -------
        Tuple[float, Array]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        if self._gradient_function is None:
            self._gradient_function = jit(grad(self._fid, argnums=0))

        states, dg_dp_list = self._propagation_and_gradient_func(times=times)
        final_state = states[-1]
        df_dp_list = []
        f = self._overlap(self._target_state, final_state)
        for dg_dp in dg_dp_list[-1]:
            g = self._overlap(self._target_state, dg_dp)
            dfdp = self._gradient_function(f) * g
            df_dp_list.append(jnp.real(dfdp))
        return StateTransferFidelity._fid(f), jnp.array(df_dp_list)  # shape scalar, (n_parameters,)


class StateTransferFidelityGRAPE(StateTransferFidelity):
    """Fidelity measure that compares overlap of the initial and final state.

    For GRAPE the optimizable parameters are vector quantities.

    Parameters
    ----------
    propagation : DifferentiablePropagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    target_state : Array
        Target state.
    times : Array
        One-dimensional vector of timestamps.

    """

    _propagation: DifferentiablePropagation

    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array] | tuple[float, Array]:
        """Compute function value and corresponding gradient.

        Returns
        -------
        Tuple[Array, Array]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        states, grads = self._propagation_and_gradient_func(times=times)
        final_state = states[-1]
        f = self._overlap(self._target_state, final_state)
        grads = 0.5 * jnp.real(f.conj() * grads + grads.conj() * f).flatten()
        return StateTransferFidelity._fid(f), grads  # shape scalar, (n_parameters,)
