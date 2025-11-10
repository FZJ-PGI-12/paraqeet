"""Class definition of the unitary fidelity model."""

import jax
import jax.numpy as jnp

from paraqeet.differentiable import Differentiable
from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array

jax.config.update("jax_enable_x64", True)


class UnitaryFidelity(NormalizableMeasurement, Differentiable):
    """Unitary fidelity measurement model.

    Fidelity measure that compares the propagator with a desired gate
    by way of L2 norm.

    Parameters
    ----------
    propagation : Propagation
        Implementation of EOM solver.
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

    __basis_states: Array | None
    __target_costates: Array
    __propagation: Propagation

    def __init__(
        self,
        propagation: Propagation,
        gate: Array,
        basis_states: Array | None = None,
    ):
        self.__propagation = propagation
        if basis_states is not None:
            self.__propagation.set_initial_state(basis_states)
        else:
            basis_states = jnp.eye(gate.shape[0])
        self.__basis_states = basis_states
        self.set_ideal_gate(gate)

    @staticmethod
    def __fid(overlaps: Array) -> float:
        """Gate fidelity from state overlaps.

        Parameters
        ----------
        overlaps : List
            State overlap as a one-dimensional array.

        Returns
        -------
        float
            Gate fidelity as a single float.

        """
        return float(jnp.abs(jnp.average(overlaps)) ** 2)

    def measure(self, times: Array) -> Array | float:
        """Return measurement in the range [0, 1]."""
        return self.calculate_normalized_scalar(times=times)

    def calculate_normalized_scalar(self, times: Array | float) -> float:
        """Return the L2 norm of the last time step compared to the ideal gate.

        Returns
        -------
        Array
            L2 norm of the last time step compared to the ideal gate.

        """
        states = self.__propagation.propagate(time=times)
        overlaps = []
        for ii, s in enumerate(self.__target_costates.T):
            overlaps.append(jnp.vdot(s, states[-1][:, ii]))
        return self.__fid(jnp.asarray(overlaps))

    def value_and_gradient(self, times: Array) -> tuple[Array, Array] | tuple[float, Array]:
        """Get the L2 norm and the analytic expression for the gradient.

        Returns
        -------
        Tuple[Array Array]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        states, dg_dp_list = self.__propagation.gradient(time=times)  # gradient of states wrt parameters
        overlaps = []
        for ii, s in enumerate(self.__target_costates.T):
            overlaps.append(jnp.vdot(s, states[-1][:, ii]))
        f = jnp.average(jnp.asarray(overlaps))

        df_dp_list = []
        for dg_dp in dg_dp_list[-1]:
            gs = []
            for ii, s in enumerate(self.__target_costates.T):
                gs.append(jnp.vdot(s, dg_dp[:, ii]))
            g = jnp.average(jnp.asarray(gs))
            df_dp_list.append(jnp.real(f.conj() * g + f * g.conj()))  # chain rule for abs^2

        fid = self.__fid(jnp.asarray(overlaps))
        return fid, jnp.array(df_dp_list)  # shape scalar, (n_parameters,)

    def set_ideal_gate(self, gate: Array):
        """Compute target states for the L2 norm.

        Parameters
        ----------
        gate : Array
            Target state computation via this gate.

        """
        if self.__basis_states is None:
            self.__target_costates = gate
        else:
            self.__target_costates = self.__basis_states @ gate
