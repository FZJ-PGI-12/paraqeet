"""The class definition of state transfer fidelity model."""

from collections.abc import Callable
from typing import override

import jax
import jax.numpy as jnp

from paraqeet.autograd_utils import get_jacobian_func
from paraqeet.differentiable import Differentiable
from paraqeet.exceptions import ConfigurationException
from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.measurement.utils import gate_fidelity
from paraqeet.quantity import Array, Float

jax.config.update("jax_enable_x64", True)


class Fidelity(NormalizableMeasurement, Differentiable):
    """Fidelity measure that compares the overlap of the propagated final state and the target state.

    TODO: Update docstring here

    This class takes the overlap function as input, in the form ``overlap(final_state, target_state, *args, **kwargs)``.
    The overlap function is assumed to be a JAX jit compatible functionally pure function.

    The fidelity function has a default implementation of ``abs(overlap)^2``.
    The user can replace the fidelity function with a JAX jit compatible function
    of the form ``fid(overlap: Array, *args, **kwargs) -> float``.

    The gradient of the ``_overlap`` and the ``_fid`` functions are computed by automatic differentiation.
    """

    _target_states: Array
    _fid: Callable[[Array], Array]
    _overlap: Callable[[Array, Array], Array]
    _propagation_func: Callable[[Array], Array]
    _propagation_gradient_func: Callable[[Array], Array]
    _overlap_grad: Callable
    _fid_grad: Callable

    def __init__(
        self,
        propagation_func: Callable[[Array], Array],
        propagation_gradient_func: Callable[[Array], Array],
        overlap: Callable[[Array, Array], Array],
        fid: Callable[[Array], Array],
        target_states: Array | None = None,
        basis_states: Array | None = None,
        ideal_gate: Array | None = None,
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
        if target_states is not None and ideal_gate is not None:
            raise ConfigurationException("Supply either target_states or and ideal_gate with basis_states")
        if ideal_gate is not None and target_states is not None:
            raise ConfigurationException("You need to supply either an ideal_gate or target_states directly.")
        self._propagation_func = propagation_func
        self._propagation_gradient_func = propagation_gradient_func
        self._target_states = jnp.array(target_states, dtype=jnp.complex128)
        if ideal_gate is not None:
            basis_states = basis_states or jnp.eye(ideal_gate.shape[0])
            self.set_ideal_gate(ideal_gate, basis_states)
        self._overlap = overlap
        self._fid = fid
        self._overlap_grad = get_jacobian_func(self._overlap)
        self._fid_grad = get_jacobian_func(fid)

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
    def get_value(self, times: Array) -> Float:
        states = self._propagation_func(jnp.array(times))
        final_state = states[-1]
        return float(self._fid(self._overlap(final_state, self._target_states)))

    @override
    def get_gradient(self, times: Array) -> Array:
        """Wrapper to satisfy the interface. DEPRECATED."""
        return self.get_value_and_gradient(times)[1]

    @override
    def get_value_and_gradient(self, times: Array) -> tuple[Array | Float, Array]:
        """Compute the gradient.

        Args:
            times: One-dimensional vector of timestamps.

        Returns:
            The gradient of shape (n_params,).
        """
        states = self._propagation_func(times)
        final_states = states[-1]
        dg_dp_list = self._propagation_gradient_func(times)  # gradient of states wrt parameters
        df_dp_list = []
        f = self._overlap(final_states, self._target_states)
        for dg_dp in dg_dp_list[-1]:
            dfdp = self._fid_grad(f) * (self._overlap_grad(dg_dp, self._target_states).T @ dg_dp)
            df_dp_list.append(jnp.real(jnp.squeeze(dfdp)))
        return self._fid(f), jnp.array(df_dp_list)  # (n_parameters,)

    def set_ideal_gate(self, gate: Array, basis_states: Array | None) -> None:
        """Compute target states by applying an ideal gate to a set of basis states.

        Args:
            gate: Target state computation via this gate.
        """
        if basis_states is None:
            self._target_states = gate
        else:
            self._target_states = basis_states @ gate


class FidelityGRAPE(Fidelity):
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
        f = self._overlap(final_state, self._target_states)
        grads = jnp.real(self._fid_grad(f) * grads)
        return self._fid(f), grads.flatten()  # shape scalar, (n_parameters,)

    @override
    def get_gradient(self, times: Array) -> Array:
        _, gradient = self.get_value_and_gradient(times)
        return gradient


class UnitaryFidelity(Fidelity):
    """Fidelity measure that compares overlap of unitary matrices.

    It compares the propagator with a desired gate by using the L2 norm, based on the
    average gate fidelity formula :cite:p:`nielsen2002simple`.
    """

    _target_states: Array
    _propagation_func: Callable[[Array], Array]
    _propagation_gradient_func: Callable[[Array], Array]

    def __init__(
        self,
        propagation_func: Callable[[Array], Array],
        propagation_gradient_func: Callable[[Array], Array],
        gate: Array,
        basis_states: Array | None = None,
    ) -> None:
        """
        Args:
            propagation_func: Function that evaluates the propagation of some
                initial state. Expected to be of the form
                ``func(t: Array) -> states: Array``.
            propagation_gradient_func: Function returning the gradients of the
                propagated states.
            gate: Matrix representation of target gate.
            basis_states: List of basis states. If set the ideal and actual
                gate are applied to these states and their pairwise overlap
                computed, equivalent to the L2 trace norm.

        """
        self._propagation_func = propagation_func
        self._propagation_gradient_func = propagation_gradient_func
        basis_states = basis_states if basis_states is not None else jnp.eye(gate.shape[0])
        self.set_ideal_gate(gate, basis_states)
        self._fid = gate_fidelity

    @override
    def get_value(self, times: Array) -> Float:
        states = self._propagation_func(jnp.array(times))
        overlaps = []
        for ii, s in enumerate(self._target_states.T):
            overlaps.append(jnp.vdot(s, states[-1][:, ii]))
        return float(self._fid(jnp.asarray(overlaps)))

    @override
    def calculate_normalized_scalar(self, times: Array) -> Float:
        """Return the L2 norm of the last time step compared to the ideal gate.

        Args:
            times: Array of times.

        Returns:
            L2 norm of the last time step compared to the ideal gate.
        """
        return self.get_value(times)

    @override
    def get_value_and_gradient(self, times: Array) -> tuple[Float | Array, Array]:
        """Get the analytic expression for the measurement value and its gradient.

        Args:
            times: Array of times.

        Returns:
            Tuple of function value and gradient of shape (n_params,).
        """
        states = self._propagation_func(times)
        dg_dp_list = self._propagation_gradient_func(times)  # gradient of states wrt parameters
        overlaps = []
        for ii, s in enumerate(self._target_states.T):
            overlaps.append(jnp.vdot(s, states[-1][:, ii]))
        f = jnp.average(jnp.asarray(overlaps))

        df_dp_list = []
        for dg_dp in dg_dp_list[-1]:
            gs = []
            for ii, s in enumerate(self._target_states.T):
                gs.append(jnp.vdot(s, dg_dp[:, ii]))
            g = jnp.average(jnp.asarray(gs))
            # TODO: Convert to AD and use this implementation as check
            df_dp_list.append(jnp.real(f.conj() * g + f * g.conj()))  # chain rule for abs^2

        return self._fid(jnp.asarray(overlaps)), jnp.array(df_dp_list)  # shape scalar, (n_parameters,)

    @override
    def get_gradient(self, times: Array) -> Array:
        _, grad = self.get_value_and_gradient(times)
        return grad
