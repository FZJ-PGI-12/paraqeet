from typing import Tuple, List

import numpy as np

from cthree.Quantity import Quantity
from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation
from cthree.Exceptions import IncompatibleLayersException


class MakhlinDistance(Measurement):
    """
    Measures the distance of a propagator to a perfect entangler using Makhlin invariants. If a list of ideal Makhlin
    invariants is given, the distance is measured as the Euclidean distance between the actual and ideal invariants.
    Else, the Makhlin distance is used.
    """

    __propagation: Propagation
    __idealInvariants: np.ndarray

    def __init__(self, propagation: Propagation, times: np.ndarray, idealInvariants: np.ndarray = None):
        super().__init__(times=times)
        self.__propagation = propagation
        self.__idealInvariants = idealInvariants

    def getParameters(self) -> List[Quantity]:
        return []

    def measure(self) -> float:
        U = self.__propagation.propagate(self._times)[-1]
        if len(U.shape) < 2 or U.shape[0] != U.shape[1]:
            raise IncompatibleLayersException(
                "quadratic unitary needed for Makhlin invariants"
            )
        gs = self.__makhlinInvariants(U)
        if self.__idealInvariants:
            return np.linalg.norm(gs - self.__idealInvariants)
        else:
            return np.abs(gs[2] * np.sqrt(gs[0] ** 2 + gs[1] ** 2) - gs[0])

    def __makhlinInvariants(self, U: np.ndarray) -> Tuple[float, float, float]:
        """
        Computes the Makhlin invariants for a matrix U. Returns a tuple with the three invariants g1,g2,g3.
        """
        # transform to bell basis
        Q = np.matrix(
            [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]
        ) / np.sqrt(2)
        Ub = Q.H @ U @ Q

        # calculate characteristics
        m = Ub.T @ Ub
        tr = np.trace(m)
        tr2 = np.trace(m**2)
        trSq = tr**2
        g1 = np.real(trSq) / 16.0
        g2 = np.imag(trSq) / 16.0
        g3 = np.real((trSq - tr2)) / 4.0
        return g1, g2, g3
