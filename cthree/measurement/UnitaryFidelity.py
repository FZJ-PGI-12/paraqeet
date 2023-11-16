import numpy as np

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
        self.__basis_states = basis_states
        self.__times = times
        self.setIdealGate(gate)

    def measure(self) -> float:
        """
        Return the L2 norm of the last time step compared to the ideal gate.
        """
        final_states = self.__propagation.propagate(time=self.__times)
        overlap = np.trace(self.__target_costates.conj().T @ final_states[-1])
        return np.abs(overlap / final_states.shape[0]) ** 2

    def setIdealGate(self, gate):
        """
        Compute target states for the L2 norm.
        """
        if self.__basis_states is None:
            self.__target_costates = gate
        else:
            self.__target_costates = gate @ self.__basis_states.conj().T
