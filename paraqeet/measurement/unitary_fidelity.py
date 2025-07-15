"""Class definition of the unitary fidelity model."""

import jax.numpy as jnp

from paraqeet.measurement.measurement import Measurement
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Quantity, Array

import jax

jax.config.update("jax_enable_x64", True)


class UnitaryFidelity(Measurement):
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
        times: Array,
        basis_states: Array | None = None,
    ):
        super().__init__(times)
        self.__propagation = propagation
        if basis_states is not None:
            self.__propagation.set_initial_state(basis_states)
        else:
            basis_states = jnp.eye(gate.shape[0])
        self.__basis_states = basis_states
        self.set_ideal_gate(gate)

    def get_parameters(self) -> list[Quantity]:
        """Get parameters of the system.

        Returns
        -------
        list[Quantity]
            Returns the parameters of the system.

        """
        return []

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

    def measure_normalised_scalar(self) -> float:
        """Return the L2 norm of the last time step compared to the ideal gate.

        Returns
        -------
        Array
            L2 norm of the last time step compared to the ideal gate.

        """
        states = self.__propagation.propagate(time=self._times)
        states = self._preprocess_matrix(states)
        overlaps = []
        for ii, s in enumerate(self.__target_costates.T):
            overlaps.append(jnp.vdot(s, states[-1][:, ii]))
        return self.__fid(jnp.asarray(overlaps))

    def measure_with_gradient(self) -> tuple[float, Array]:
        """Get the L2 norm and the analytic expression for the gradient.

        Returns
        -------
        Tuple[Array Array]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        states, dg_dp_list = self.__propagation.gradient(time=self._times)  # gradient of states wrt parameters
        states = self._preprocess_matrix(states)
        dg_dp_list = self._preprocess_matrix(dg_dp_list)
        overlaps = []
        for ii, s in enumerate(self.__target_costates.T):
            overlaps.append(jnp.vdot(s, states[-1][:, ii]))
        f = jnp.average(jnp.asarray(overlaps))

        dF_dp = []
        for dg_dp in dg_dp_list[-1]:
            gs = []
            for ii, s in enumerate(self.__target_costates.T):
                gs.append(jnp.vdot(s, dg_dp[:, ii]))
            g = jnp.average(jnp.asarray(gs))
            dF_dp.append(jnp.real(f.conj() * g + f * g.conj()))  # chain rule for abs^2

        fid = self.__fid(jnp.asarray(overlaps))
        return fid, jnp.array(dF_dp)  # shape scalar, (n_parameters,)

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
