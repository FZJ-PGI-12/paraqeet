"""Class definition of the Makhlin functional."""

from collections.abc import Callable

import jax.numpy as jnp

from paraqeet.exceptions import (
    ConfigurationException,
    IncompatibleLayersException,
)
from paraqeet.measurement.measurement import Measurement
from paraqeet.quantity import Array, Float


class MakhlinFunctional(Measurement):
    """Class definition of the Makhlin Functional invariants.

    Measures the distance of a propagator to a perfect entangler
    using Makhlin invariants.
    If a list of ideal Makhlin invariants is given,
    the distance is measured as the Euclidean distance between
    the actual and ideal invariants.
    Else, the Makhlin distance is used.
    """

    _propagation_func: Callable[[Array], Array]
    _ideal_invariants: Array | None

    def __init__(
        self,
        propagation_func: Callable[[Array], Array],
        ideal_invariants: Array | None = None,
    ) -> None:
        """
        Args:
            propagation_func: Function that evaluates the propagation of some
                initial state. Expected to be of the form
                ``func(t: Array) -> states: Array``.
            ideal_invariants: One-dimensional vector of ideal Makhlin
                invariants.
        """
        self._propagation_func = propagation_func
        self._ideal_invariants = ideal_invariants

    def measure(self, times: Array) -> Array | Float:
        """Measure distance of the propagator to a perfect entangler.

        Args:
            times: One-dimensional vector of timestamps.

        Returns:
            Distance of propagator.

        Raises:
            IncompatibleLayersException: If a quadratic unitary 4x4 operator
                is not received.
        """
        if not times:
            raise ConfigurationException("Time array was not specified")

        u = self._propagation_func(times)[-1]
        if u.shape != (4, 4):
            raise IncompatibleLayersException("quadratic unitary 4x4 propagator needed for Makhlin invariants")
        gs = self._makhlin_invariants(u)
        if self._ideal_invariants is not None:
            return jnp.array(jnp.linalg.norm(gs - self._ideal_invariants))
        else:
            return jnp.abs(gs[2] * jnp.sqrt(gs[0] ** 2 + gs[1] ** 2) - gs[0])

    def _makhlin_invariants(self, u: Array) -> tuple[Array, Array, Array]:
        """Compute the Makhlin invariants for a matrix U.

        Returns a tuple with the three invariants g1, g2 and g3.

        Args:
            u: Input matrix for computing the Makhlin invariants of.

        Returns:
            Tuple of 3 Arrays as invariants g1, g2 and g3.
        """
        # transform to bell basis
        q = jnp.array(
            [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
        )
        det = jnp.linalg.det(u)
        # Normalize the determinant to be sensitive to leakage, non-unitarity.
        if det != 0.0:
            det /= jnp.abs(det)
        u_b = (q.T.conj() @ u @ q) / 2
        m = u_b.T @ u_b
        tr = jnp.trace(m @ m)
        tr_sq = jnp.trace(m) ** 2 / det
        return (
            jnp.real(tr_sq) / 16,
            jnp.imag(tr_sq) / 16,
            jnp.real(tr_sq - tr / det) / 4,
        )
