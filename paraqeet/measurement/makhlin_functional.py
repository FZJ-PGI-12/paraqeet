"""Class definition of the Makhlin functional."""

import jax.numpy as jnp
from paraqeet.quantity import Array

from paraqeet.exceptions import ConfigurationException, IncompatibleLayersException
from paraqeet.measurement.measurement import Measurement
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Quantity


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
    propagation : Propagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    times : Array
        One-dimensional vector of timestamps.
    ideal_invariants : Array optional
        One-dimensional vector of ideal Makhlin invariants.

    """

    __propagation: Propagation
    __ideal_invariants: Array | None

    def __init__(
        self,
        propagation: Propagation,
        times: Array,
        ideal_invariants: Array | None = None,
    ):
        super().__init__(times=times)
        self.__propagation = propagation
        self.__ideal_invariants = ideal_invariants

    def get_parameters(self) -> list[Quantity]:
        """Get the parameters of the system.

        Returns
        -------
        list[Quantity]
            Returns the list of parameters of the system.

        """
        return []

    def measure(self) -> Array:
        """Measure distance of the propagator to a perfect entangler.

        Returns
        -------
        Array
            Distance of propagator.

        Raises
        ------
        IncompatibleLayersException
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
            return jnp.array(jnp.linalg.norm(gs - self.__ideal_invariants))
        else:
            return jnp.abs(gs[2] * jnp.sqrt(gs[0] ** 2 + gs[1] ** 2) - gs[0])

    def __makhlin_invariants(self, U: Array) -> tuple[Array, Array, Array]:
        """Compute the Makhlin invariants for a matrix U.

        Returns a tuple with the three invariants g1, g2 and g3.

        Parameters
        ----------
        U: Array
            Input matrix for computing the Makhlin invariants of.

        Returns
        -------
        Tuple[Array, Array, Array]
            Returns a tuple of 3 Numpy Array as invariants g1, g2 and g3.

        """
        # transform to bell basis
        Q = jnp.array(
            [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
        )
        det = jnp.linalg.det(U)
        # Normalize the determinant to be sensitive to leakage, non-unitarity.
        if det != 0.0:
            det /= jnp.abs(det)
        U_B = (Q.T.conj() @ U @ Q) / 2
        m = U_B.T @ U_B
        tr = jnp.trace(m @ m)
        trSq = jnp.trace(m) ** 2 / det
        return (
            jnp.real(trSq) / 16,
            jnp.imag(trSq) / 16,
            jnp.real(trSq - tr / det) / 4,
        )
