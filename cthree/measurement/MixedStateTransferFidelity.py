"""Class definition for a mixed state transfer fidelity model."""

import numpy as np
import scipy.linalg as sclin

from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation
from cthree.Exceptions import IncompatibleLayersException


class MixedStateTransferFidelity(Measurement):
    """Mixed state transfer fidelity measurement model.

    Fidelity measure that compares the overlap of the initial
    and final state of density matrices.
    Note: this implementation is still very inaccurate.

    Parameters
    ----------
    propagation : cthree.propagation.Propagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    targetState : numpy.ndarray
        Final state of the density matrices.
    times : numpy.ndarray
        One-dimensional vector of timestamps.

    """

    __targetState: np.ndarray
    __targetStateSqrt: np.ndarray | None
    __propagation: Propagation
    __times: np.ndarray

    def __init__(
        self,
        propagation: Propagation,
        targetState: np.ndarray,
        times: np.ndarray,
    ):
        super().__init__()
        self.__propagation = propagation
        self.__targetState = targetState
        self.__times = times

        # store the sqrt of the density matrix to simplify the measurement
        self.__targetStateSqrt = sclin.sqrtm(self.__targetState)

    def measure(self) -> np.ndarray:
        """Measure overlap between initial and final state of density matrices.

        Returns
        -------
        numpy.ndarray
            Overlap between initial and final state of density matrices.

        Raises
        ------
        cthree.Exceptions.IncompatibleLayersException
            Raises an exception if required vector shape is not received.

        """
        state = self.__propagation.propagate(self.__times)[-1]
        state = self._preprocessMatrix(state)
        if state.shape != self.__targetState.shape:
            raise IncompatibleLayersException(
                f"Need a state vector of size {self.__targetState.shape}"
                "for the state transfer fidelity, "
                "but got shape {state.shape}"
            )

        # density matrix
        product = self.__targetStateSqrt @ state @ self.__targetStateSqrt
        return np.abs(np.trace(sclin.sqrtm(product))) ** 2
