"""Class definition of the Makhlin functional."""

import jax.numpy as np

from cthree.quantity import Quantity, yaqArray
from cthree.measurement.measurement import Measurement
from cthree.propagation.propagation import Propagation
from cthree.exceptions import IncompatibleLayersException, ConfigurationException


class MakhlinFunctional(Measurement):
    """Class definition of the Makhlin Functional invariants.

    Measures the distance of a propagator to a perfect entangler
    using Makhlin invariants.
    If a list of ideal Makhlin invariants is given,
    the distance is measured as the Euclidean distance between
    the actual and ideal invariants.
    Else, the Makhlin distance is used.

    Parameters
    ----------
    propagation : cthree.propagation.propagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    times : numpy.ndarray
        One-dimensional vector of timestamps.
    ideal_invariants : numpy.ndarray, optional
        One-dimensional vector of ideal Makhlin invariants.

    """

    __propagation: Propagation
    __ideal_invariants: np.ndarray | None

    def __init__(
        self,
        propagation: Propagation,
        times: np.ndarray,
        ideal_invariants: np.ndarray | None = None,
    ):
        super().__init__(times=times)
        self.__propagation = propagation
        self.__ideal_invariants = ideal_invariants

    def get_parameters(self) -> list[Quantity]:
        """Get the parameters of the system.

        Returns
        -------
        list[cthree.quantity]
            Returns the list of parameters of the system.

        """
        return []

    def measure(self) -> yaqArray:
        """Measure distance of the propagator to a perfect entangler.

        Returns
        -------
        numpy.ndarray
            Distance of propagator.

        Raises
        ------
        cthree.Exceptions.IncompatibleLayersException
            Raises an exception if a quadratic unitary
            4x4 operator is not received.

        """
        if self._times is None:
            raise ConfigurationException("Time array was not specified")

        U = self.__propagation.propagate(self._times)[-1]
        U = self._preprocess_matrix(U)
        if U.shape != (4, 4):
            raise IncompatibleLayersException("quadratic unitary 4x4 propagator needed for Makhlin invariants")
        gs = self.__makhlin_invariants(U)
        if self.__ideal_invariants is not None:
            return np.linalg.norm(gs - self.__ideal_invariants)  # type: ignore
        else:
            return np.abs(gs[2] * np.sqrt(gs[0] ** 2 + gs[1] ** 2) - gs[0])

    def __makhlin_invariants(self, U: yaqArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the Makhlin invariants for a matrix U.

        Returns a tuple with the three invariants g1, g2 and g3.

        Parameters
        ----------
        U: numpy.ndarray
            Input matrix for computing the Makhlin invariants of.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Returns a tuple of 3 Numpy ndarrays as invariants g1, g2 and g3.

        """
        # transform to bell basis
        Q = np.array(
            [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
        )
        det = np.linalg.det(U)
        # Normalize the determinant to be sensitive to leakage, non-unitarity.
        if det != 0.0:
            det /= np.abs(det)
        U_B = (Q.T.conj() @ U @ Q) / 2
        m = U_B.T @ U_B
        tr = np.trace(m @ m)
        trSq = np.trace(m) ** 2 / det
        return (
            np.real(trSq) / 16,
            np.imag(trSq) / 16,
            np.real(trSq - tr / det) / 4,
        )
