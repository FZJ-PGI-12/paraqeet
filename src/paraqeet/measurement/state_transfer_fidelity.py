"""The class definition of state transfer fidelity model."""

from collections.abc import Callable
from typing import override

import jax
import jax.numpy as jnp

from paraqeet.autograd_utils import get_jacobian_func
from paraqeet.differentiable import Differentiable
from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.quantity import Array, Float

jax.config.update("jax_enable_x64", True)


class StateTransferFidelity(NormalizableMeasurement, Differentiable):
    """Fidelity measure that compares the overlap of the propagated final state and the target state.

    This class takes the overlap function as input, in the form ``overlap(final_state, target_state, *args, **kwargs)``.
    The overlap function is assumed to be a JAX jit compatible functionally pure function.

    The fidelity function has a default implementation of ``abs(overlap)^2``.
    The user can replace the fidelity function with a JAX jit compatible function
    of the form ``fid(overlap: Array, *args, **kwargs) -> float``.

    The gradient of the ``_overlap`` and the ``_fid`` functions are computed by automatic differentiation.
    """

    _target_state: Array
    _overlap: Callable[[Array, Array], Array]
    _propagation_func: Callable[[Array], Array]
    _propagation_gradient_func: Callable[[Array], Array]
    _overlap_grad: Callable
    _fid_grad: Callable

    def __init__(
        self,
        propagation_func: Callable[[Array], Array],
        propagation_gradient_func: Callable[[Array], Array],
        target_state: Array,
        overlap: Callable[[Array, Array], Array],
    ) -> None:
        """
        Args:
            propagation_func: Function that evaluates the propagation of some
                initial state. Expected to be of the form
                ``func(t: Array) -> states: Array``.
            propagation_gradient_func: Function returning the gradient of the
                propagated states.
            target_state: Target state.
            overlap: Overlap function of the form
                ``overlap(final_state, target_state)``.
        """
        self._propagation_func = propagation_func
        self._propagation_gradient_func = propagation_gradient_func
        self._target_state = jnp.array(target_state, dtype=jnp.complex128)
        self._overlap = overlap
        self._overlap_grad = get_jacobian_func(self._overlap)
        self._fid_grad = get_jacobian_func(self._fid)

    @staticmethod
    def _fid(overlap: Array) -> Float:
        return (jnp.abs(jnp.average(overlap)) ** 2).astype(float)

    @override
    def get_value(self, times: Array) -> Float:
        states = self._propagation_func(jnp.array(times))
        final_state = states[-1]
        return self._fid(self._overlap(final_state, self._target_state))

    @override
    def calculate_normalized_scalar(self, times: Array) -> Float:
        """Measure the fidelity between the propagated final state and the target state. To be used with an optimizer.

        For NormalizableMeasurement objects that are also Differentiable this coincides
        with the get_value method.

        Args:
            times: One-dimensional vector of timestamps.

        Returns:
            Fidelity between the final and target state as a bare Float.
        """
        return self.get_value(times)

    @override
    def get_gradient(self, times: Array) -> Array:
        """Compute the gradient.

        Args:
            times: One-dimensional vector of timestamps.

        Returns:
            The gradient of shape (n_params,).
        """
        states, dg_dp_list = self._propagation_gradient_func(times)
        final_state = states[-1]
        df_dp_list = []
        f = self._overlap(final_state, self._target_state)
        for dg_dp in dg_dp_list[-1]:
            dfdp = self._fid_grad(f) * (self._overlap_grad(dg_dp, self._target_state).T @ dg_dp)
            df_dp_list.append(jnp.real(jnp.squeeze(dfdp)))
        return jnp.array(df_dp_list)  # (n_parameters,)


class StateTransferFidelityGRAPE(StateTransferFidelity):
    """Fidelity measure that compares the overlap of the propagated final state and the target state.

    For GRAPE the optimizable parameters are vector quantities given by the PWC bins of the pulse.

    This class takes the overlap function as input, in the form ``overlap(final_state, target_state, *args, **kwargs)``.
    The overlap function is assumed to be a JAX jit compatible functionally pure function.

    The fidelity function has a default implementation of ``abs(overlap)^2``.
    The user can replace the fidelity function with a JAX jit compatible function
    of the form ``fid(overlap: Array, *args, **kwargs) -> float``.

    The gradient of the ``_overlap`` and the ``_fid`` functions are computed by automatic differentiation.
    """

    @override
    def get_value_and_gradient(self, times: Array) -> tuple[Array | Float, Array]:
        """Compute function value and corresponding gradient.

        Returns:
            Tuple of function value and gradient of shape (n_parameters,).
        """
        states = self._propagation_func(times)
        grads = self._propagation_gradient_func(times)
        final_state = states[-1]
        f = self._overlap(final_state, self._target_state)
        grads = jnp.real(self._fid_grad(f) * grads)
        return self._fid(f), grads.flatten()  # shape scalar, (n_parameters,)

    @override
    def get_gradient(self, times: Array) -> Array:
        _, gradient = self.get_value_and_gradient(times)
        return gradient
