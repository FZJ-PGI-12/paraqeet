import numpy as np

from typing import List, Tuple

from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation


class UnitaryFidelity(Measurement):
    """
    Fidelity measure that compares the propagator with a desired gate by way of L2 norm.

    Args:
        propagation (Propagation): Implementation of EOM solver
        gate (np.ndarray): Matrix representation of target gate
        times (List[float]): List of times to compare. Should have length 2. More is
            allowed, but only the first and last are used.
        basis_states (List[np.ndarray], optional): List of basis states. If set the ideal
            and actual gate are applied to these states and their pairwise overlap computed,
            equivalent to the L2 trace norm. Defaults to [].
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
        self.__bestFid = 0
        self.setIdealGate(gate)

    @staticmethod
    def __fid(overlaps: List) -> np.ndarray:
        """
        Gate fidelity from state overlaps.
        """
        return np.abs(np.average(overlaps)) ** 2

    def measure(self) -> np.ndarray:
        """
        Return the L2 norm of the last time step compared to the ideal gate.
        """
        states = self.__propagation.propagate(time=self._times)
        overlaps = []
        for ii, s in enumerate(self.__target_costates.T):
            overlaps.append(np.vdot(s, states[-1][:, ii]))
        fid = self.__fid(overlaps)
        if fid > self.__bestFid:
            self.__bestFid = fid
            self._bestState = self.__basis_states.T @ states[-1]
        return self.__fid(overlaps)

    def measureWithGradient(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gives the L2 norm and the analytic expression for the gradient.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of function value and gradient of shape (n_parameters,)
        """
        states, dg_dp_list = self.__propagation.gradient(
            time=self._times
        )  # gradient of states wrt parameters
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
        if fid > self.__bestFid:
            self.__bestFid = fid
            self._bestState = self.__basis_states.T @ states[-1]
        return fid, np.array(dF_dp)  # shape scalar, (n_parameters,)

    def setIdealGate(self, gate):
        """
        Compute target states for the L2 norm.
        """
        if self.__basis_states is None:
            self.__target_costates = gate
        else:
            self.__target_costates = self._UnitaryFidelity__basis_states @ gate
        self.__bestFid = 0
