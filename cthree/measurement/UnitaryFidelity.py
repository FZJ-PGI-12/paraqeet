import numpy as np

from typing import List

from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation


class UnitaryFidelity(Measurement):
    """
    Fidelity measure that compares the propagator with a desired gate by way of L2 norm.
    """

    __basis_states: np.ndarray
    __target_costates: np.ndarray
    __propagation: Propagation

    def __init__(
        self,
        propagation: Propagation,
        gate: np.ndarray,
        basis_states: List[np.ndarray],
        times: List[float],
    ):
        super().__init__()
        self.__propagation = propagation
        self.__propagation.setInitialStates(basis_states)
        self.__basis_states = basis_states
        self.__times = times
        self.setIdealGate(gate)

    def measure(self) -> float:
        final_states = self.__propagation.propagate(time=self.__times)
        overlap = np.trace(self.__target_costates @ final_states[-1])
        return np.abs(overlap / len(self.__basis_states)) ** 2

    def setIdealGate(self, gate):
        """
        Compute target states for the L2 norm.
        """
        self.__target_costates = gate @ np.concatenate(self.__basis_states, axis=1).T
