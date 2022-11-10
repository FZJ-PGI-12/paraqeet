from typing import Tuple

import numpy as np

from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation
from cthree.Exceptions import IncompatibleLayersException


class MakhlinDistance(Measurement):
    """
    Measures the distance of a propagator to a perfect entangler using Makhlin invariants.
    """
    __propagation: Propagation

    def __init__(self, propagation: Propagation, initialState: np.array):
        super().__init__()
        self.__propagation = propagation

    def measure(self) -> float:
        U = self.__propagation.propagate()
        if U.shape[0] != U.shape[1]:
            raise IncompatibleLayersException(f"quadratic unitary needed for Makhlin invariants")
        gs = self.__makhlinInvariants(U)
        return self.__makhlinDistance(*gs)

    def __makhlinInvariants(self, U: np.array) -> Tuple[float, float, float]:
        """
        Computes the Makhlin invariants for a matrix U. Returns a tuple with the three invariants g1,g2,g3.
        """
        # transform to bell basis
        Q = np.matrix([
            [1, 0, 0, 1j],
            [0, 1j, 1, 0],
            [0, 1j, -1, 0],
            [1, 0, 0, -1j]
        ]) / np.sqrt(2)
        Ub = Q.H @ U @ Q

        # calculate characteristics
        m = Ub.T @ Ub
        tr = np.trace(m)
        tr2 = np.trace(m ** 2)
        trSq = tr ** 2
        g1 = np.real(trSq) / 16.0
        g2 = np.imag(trSq) / 16.0
        g3 = np.real((trSq - tr2)) / 4.0
        return g1, g2, g3

    def __makhlinDistance(self, g1: float, g2: float, g3: float) -> float:
        """
        Computes the distance of the point specified by three invariants to the space of perfect entanglers.
        """
        roots = np.roots([1, -g3, 4 * np.sqrt(g1 ** 2 + g2 ** 2) - 1, g3 - 4 * g1]).real()
        roots = np.round(roots, 5)
        z = np.sort(roots)

        d = g3 * np.sqrt(g1 ** 2 + g2 ** 2) - g1
        s = np.pi - np.arccos(z[0]) - np.arccos(z[2])
        if d > 0 and s > 0:
            return d
        elif d < 0 and s < 0:
            return -d
        else:
            return 0
