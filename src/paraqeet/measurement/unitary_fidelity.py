"""Class definition of the unitary fidelity model."""

from collections.abc import Callable
from typing import override

import jax
import jax.numpy as jnp

from paraqeet.differentiable import Differentiable
from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.quantity import Array, Float

jax.config.update("jax_enable_x64", True)


class UnitaryFidelity(NormalizableMeasurement, Differentiable):
    """Unitary fidelity measurement model.

    Fidelity measure that compares the propagator with a desired gate
    by way of L2 norm.

    Parameters
    ----------
    propagation_func: Callable[[Array], Array]
        Function that evaluates the propagation of some initial state.
        Expected to be of the form `func(t: Array) -> states: Array`.
    propagation_and_gradient_func: Callable[[Array], tuple[Array, Array]]
        Function returning the propagated states and their gradients.
    gate : Array
        Matrix representation of target gate.
    times : Array
        List of times to compare. Should have length 2.
        More is allowed, but only the first and last are used.
    basis_states : Array optional
        List of basis states.
        If set the ideal and actual gate are applied to these states
        and their pairwise overlap computed, equivalent to the L2 trace norm.
        Defaults to [].

    """

    _basis_states: Array | None
    _target_costates: Array
    _propagation_func: Callable[[Array], Array]
    _propagation_and_gradient_func: Callable[[Array], tuple[Array, Array]]

    def __init__(
        self,
        propagation_func: Callable[[Array], Array],
        propagation_and_gradient_func: Callable[[Array], tuple[Array, Array]],
        gate: Array,
        basis_states: Array | None = None,
    ):
        self._propagation_func = propagation_func
        self._propagation_and_gradient_func = propagation_and_gradient_func
        self._basis_states = basis_states if basis_states is not None else jnp.eye(gate.shape[0])
        self.set_ideal_gate(gate)

    @staticmethod
    def _fid(overlaps: Array) -> Float:
        """Gate fidelity from state overlaps.

        Parameters
        ----------
        Overlaps: Array
            State overlap as a one-dimensional array.

        Returns
        -------
        Float
            Gate fidelity as a single float.

        """
        return float(jnp.abs(jnp.average(overlaps)) ** 2)

    @override
    def get_value(self, times: Array) -> Float:
        states = self._propagation_func(jnp.array(times))
        overlaps = []
        for ii, s in enumerate(self._target_costates.T):
            overlaps.append(jnp.vdot(s, states[-1][:, ii]))
        return self._fid(jnp.asarray(overlaps))

    @override
    def measure(self, times: Array) -> Float:
        """Return measurement in the range [0, 1]."""
        return self.calculate_normalized_scalar(times=times)

    @override
    def calculate_normalized_scalar(self, times: Array) -> Float:
        """Return the L2 norm of the last time step compared to the ideal gate.

        Parameters
        ----------
        times : Array
            Array of times.


        Returns
        -------
        Array
            L2 norm of the last time step compared to the ideal gate.
        """
        return self.get_value(times)

    @override
    def get_gradient(self, times: Array) -> Array:
        """Get the analytic expression for the gradient.

        Parameters
        ----------
        times : Array
            Array of times.

        Returns
        -------
        Array
            Tuple of function value and gradient of shape (n_params,).

        """
        states, dg_dp_list = self._propagation_and_gradient_func(times)  # gradient of states wrt parameters
        overlaps = []
        for ii, s in enumerate(self._target_costates.T):
            overlaps.append(jnp.vdot(s, states[-1][:, ii]))
        f = jnp.average(jnp.asarray(overlaps))

        df_dp_list = []
        for dg_dp in dg_dp_list[-1]:
            gs = []
            for ii, s in enumerate(self._target_costates.T):
                gs.append(jnp.vdot(s, dg_dp[:, ii]))
            g = jnp.average(jnp.asarray(gs))
            df_dp_list.append(jnp.real(f.conj() * g + f * g.conj()))  # chain rule for abs^2

        return jnp.array(df_dp_list)  # shape scalar, (n_parameters,)

    def set_ideal_gate(self, gate: Array):
        """Compute target states for the L2 norm.

        Parameters
        ----------
        gate : Array
            Target state computation via this gate.

        """
        if self._basis_states is None:
            self._target_costates = gate
        else:
            self._target_costates = self._basis_states @ gate
