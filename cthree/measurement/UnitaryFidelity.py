"""Class definition of the unitary fidelity model."""

import numpy as np


from cthree.Quantity import Quantity
from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation


class UnitaryFidelity(Measurement):
    """Unitary fidelity measurement model.

    Fidelity measure that compares the propagator with a desired gate
    by way of L2 norm.

    Parameters
    ----------
    propagation : cthree.propagation.Propagation
        Implementation of EOM solver.
    gate : numpy.ndarray
        Matrix representation of target gate.
    times : numpy.ndarray
        List of times to compare. Should have length 2.
        More is allowed, but only the first and last are used.
    basis_states : numpy.ndarray, optional
        List of basis states.
        If set the ideal and actual gate are applied to these states
        and their pairwise overlap computed, equivalent to the L2 trace norm.
        Defaults to [].

    """

    __basis_states: np.ndarray | None
    __target_costates: np.ndarray
    __propagation: Propagation
    _times: np.ndarray

    def __init__(
        self,
        propagation: Propagation,
        gate: np.ndarray,
        times: np.ndarray,
        basis_states: np.ndarray = None,
    ):
        super().__init__()
        self.__propagation = propagation
        if basis_states is not None:
            self.__propagation.setInitialState(basis_states)
        else:
            basis_states = np.eye(gate.shape[0])
        self.__basis_states = basis_states
        self._times = times
        self.set_ideal_gate(gate)

    def get_parameters(self) -> list[Quantity]:
        """Get parameters of the system.

        Returns
        -------
        list[cthree.Quantity]
            Returns the parameters of the system.

        """
        return []

    @staticmethod
    def __fid(overlaps: list) -> np.ndarray:
        """Gate fidelity from state overlaps.

        Parameters
        ----------
        overlaps : List
            State overlap as a one-dimensional array.

        Returns
        -------
        numpy.ndarray
            Gate fidelity as a Numpy ndarray.

        """
        return np.abs(np.average(overlaps)) ** 2

    def measure(self) -> np.ndarray:
        """Return the L2 norm of the last time step compared to the ideal gate.

        Returns
        -------
        numpy.ndarray
            L2 norm of the last time step compared to the ideal gate.

        """
        states = self.__propagation.propagate(time=self._times)
        states = self._preprocess_matrix(states)
        overlaps = []
        for ii, s in enumerate(self.__target_costates.T):
            overlaps.append(np.vdot(s, states[-1][:, ii]))
        return self.__fid(overlaps)

    def measure_with_gradient(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the L2 norm and the analytic expression for the gradient.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        states, dg_dp_list = self.__propagation.gradient(time=self._times)  # gradient of states wrt parameters
        states = self._preprocess_matrix(states)
        dg_dp_list = self._preprocess_matrix(dg_dp_list)
        overlaps = []
        for ii, s in enumerate(self.__target_costates.T):
            overlaps.append(np.vdot(s, states[-1][:, ii]))
        f = np.average(overlaps)

        dF_dp = []
        for dg_dp in dg_dp_list[-1]:
            gs = []
            for ii, s in enumerate(self.__target_costates.T):
                gs.append(np.vdot(s, dg_dp[:, ii]))
            g = np.average(gs)
            dF_dp.append(np.real(f.conj() * g + f * g.conj()))  # chain rule for abs^2

        fid = self.__fid(overlaps)
        return fid, np.array(dF_dp)  # shape scalar, (n_parameters,)

    def set_ideal_gate(self, gate: np.ndarray):
        """Compute target states for the L2 norm.

        Parameters
        ----------
        gate : numpy.ndarray
            Target state computation via this gate.

        """
        if self.__basis_states is None:
            self.__target_costates = gate
        else:
            self.__target_costates = self.__basis_states @ gate
