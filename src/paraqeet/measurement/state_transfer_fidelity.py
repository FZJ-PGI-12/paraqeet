"""The class definition of state transfer fidelity model."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from paraqeet.differentiable import Differentiable
from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.measurement.utils import vjp_jacobian
from paraqeet.propagation.propagation import DifferentiablePropagation
from paraqeet.quantity import Array, Float

jax.config.update("jax_enable_x64", True)


class StateTransferFidelity(NormalizableMeasurement, Differentiable):
    """Fidelity measure that compares overlap of the initial and final state.

    This class takes the overlap function as input, in the form `overlap(final_state, target_state, *args, **kwargs)`.
    The overlap function is assumed to be a JAX grad compatible functionally pure function.

    The fidelity function has a default implementation of `abs(overlap)^2`.
    The user can replace the fidelity function with a JAX grad compatible function
    of the form `fid(overlap: Array, *args, **kwargs) -> float`.

    The gradient of the `_overlap` and the `_fid` functions are computed by automatic differentiation.

    Parameters
    ----------
    propagation_func: Callable[[Array], Array]
        Function that evaluates the propagation of some initial state.
        Expected to be of the form `func(t: Array) -> states: Array`.
    propagation_and_gradient_func: Callable[[Array], tuple[Array, Array]]
    target_state : Array
        Target state.
    times : Array
        One-dimensional vector of timestamps.
    """

    _target_state: Array
    _overlap: Callable[[Array, Array], Array]
    _propagation_func: Callable[[Array], Array]
    _propagation_and_gradient_func: Callable[[Array], tuple[Array, Array]]
    _overlap_grad: Callable
    _fid_grad: Callable

    def __init__(
        self,
        propagation_func: Callable,
        propagation_and_gradient_func: Callable,
        target_state: Array,
        overlap: Callable[[Array, Array], Array],
    ):
        self._propagation_func = propagation_func
        self._propagation_and_gradient_func = propagation_and_gradient_func
        self._target_state = jnp.array(target_state, dtype=jnp.complex128)
        self._overlap = overlap
        self._overlap_grad = vjp_jacobian(self._overlap)
        self._fid_grad = vjp_jacobian(self._fid)

    @staticmethod
    def _fid(overlap: Array) -> Float:
        return (jnp.abs(jnp.average(overlap)) ** 2).astype(float)

    def measure(self, times: Array) -> Array | Float:
        """Return measurement in the range [0, 1]."""
        return self.calculate_normalized_scalar(times=times)

    def calculate_normalized_scalar(self, times: Array) -> Float:
        """Measure overlap between initial and target state. To be used with an optimizer.

        Parameters
        ----------
        times : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Float
            Overlap between initial and target state in a bare Float.

        """
        states = self._propagation_func(times)
        final_state = states[-1]
        f = self._overlap(final_state, self._target_state)
        return StateTransferFidelity._fid(f)

    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array] | tuple[Float, Array]:
        """Compute function value and corresponding gradient.

        Returns
        -------
        Tuple[Array, Array]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        states, dg_dp_list = self._propagation_and_gradient_func(times)
        final_state = states[-1]
        df_dp_list = []
        f = self._overlap(final_state, self._target_state)
        for dg_dp in dg_dp_list[-1]:
            dfdp = self._fid_grad(f) * (self._overlap_grad(dg_dp, self._target_state).T @ dg_dp)
            df_dp_list.append(jnp.real(jnp.squeeze(dfdp)))
        return StateTransferFidelity._fid(f), jnp.array(df_dp_list)  # shape scalar, (n_parameters,)


class StateTransferFidelityGRAPE(StateTransferFidelity):
    """Fidelity measure that compares overlap of the initial and final state.

    For GRAPE the optimizable parameters are vector quantities given by the PWC bins of the pulse.

    This class takes the overlap function as input, in the form `overlap(final_state, target_state, *args, **kwargs)`.
    The overlap function is assumed to be a JAX grad compatible functionally pure function.

    The fidelity function has a default implementation of `abs(overlap)^2`.
    The user can replace the fidelity function with a JAX grad compatible function
    of the form `fid(overlap: Array, *args, **kwargs) -> float`.

    The gradient of the `_overlap` and the `_fid` functions are computed by automatic differentiation.

    Parameters
    ----------
    propagation_func: Callable[[Array], Array]
        Function that evaluates the propagation of some initial state.
        Expected to be of the form `func(t: Array) -> states: Array`.
    propagation_and_gradient_func: Callable[[Array], tuple[Array, Array]]
    target_state : Array
        Target state.
    times : Array
        One-dimensional vector of timestamps.
    """

    _propagation: DifferentiablePropagation

    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array] | tuple[Float, Array]:
        """Compute function value and corresponding gradient.

        Returns
        -------
        Tuple[Array, Array]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        states, grads = self._propagation_and_gradient_func(times)
        final_state = states[-1]
        f = self._overlap(final_state, self._target_state)
        grads = self._fid_grad(f) * grads
        return StateTransferFidelity._fid(f), grads.flatten()  # shape scalar, (n_parameters,)
