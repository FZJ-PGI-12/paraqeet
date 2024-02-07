from typing import Tuple, List

import numpy as np

from cthree.Quantity import Quantity
from cthree.measurement.Measurement import Measurement
from cthree.propagation.Propagation import Propagation
from cthree.Exceptions import IncompatibleLayersException


class MakhlinFunctional(Measurement):
    """
    Measures the distance of a propagator to a perfect entangler using Makhlin invariants. If a list of ideal Makhlin
    invariants is given, the distance is measured as the Euclidean distance between the actual and ideal invariants.
    Else, the Makhlin distance is used.
    """

    __propagation: Propagation
    __idealInvariants: np.ndarray

    def __init__(
        self,
        propagation: Propagation,
        times: np.ndarray,
        idealInvariants: np.ndarray = None,
    ):
        super().__init__(times=times)
        self.__propagation = propagation
        self.__idealInvariants = idealInvariants

    def getParameters(self) -> List[Quantity]:
        return []

    def measure(self) -> np.ndarray:
        U = self.__propagation.propagate(self._times)[-1]
        if U.shape != (4, 4):
            raise IncompatibleLayersException(
                "quadratic unitary 4x4 propagator needed for Makhlin invariants"
            )
        gs = self.__makhlinInvariants(U)
        if self.__idealInvariants is not None:
            return np.linalg.norm(gs - self.__idealInvariants)
        else:
            return np.abs(gs[2] * np.sqrt(gs[0] ** 2 + gs[1] ** 2) - gs[0])

    def __makhlinInvariants(
        self, U: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the Makhlin invariants for a matrix U. Returns a tuple with the three invariants g1,g2,g3.
        """
        # transform to bell basis
        Q = np.matrix(
            [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
        )
        det = np.linalg.det(U)
        # Normalize the determinant to be sensitive to leakage, non-unitarity.
        if det != 0.0:
            det /= np.abs(det)
        U_B = (Q.H @ U @ Q) / 2
        m = U_B.T @ U_B
        tr = np.trace(m @ m)
        trSq = np.trace(m) ** 2 / det
        return (
            np.real(trSq) / 16,
            np.imag(trSq) / 16,
            np.real(trSq - tr / det) / 4,
        )
