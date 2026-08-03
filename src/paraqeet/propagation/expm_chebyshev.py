"""Class definition of the Chebyshev expansion of the matrix exponential propagation."""

from collections.abc import Callable
from functools import partial
from typing import Any, override

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.lax import scan, stop_gradient
from jax.scipy.special import gammaln
from scipy.special import jv  # Bessel function of first kind

from paraqeet.exceptions import ConfigurationException
from paraqeet.propagation.propagation import Propagation
from paraqeet.propagation.utils import construct_times
from paraqeet.quantity import Array, Float

jax.config.update("jax_enable_x64", True)


class ExpmChebyshev(Propagation):
    r"""Piecewise matrix exponential propagation with a Chebyshev expansion.

    Like :class:`~paraqeet.propagation.expm.Expm` this solves the equation of motion by
    exponentiating the generator of every propagation step. The exponential
    :math:`\exp(A)\,\psi`, with :math:`A` the generator scaled by the step size,
    is performed by expanding it into Chebyshev polynomials :cite:p:`talezer1984accurate`.
    Writing :math:`A = -iB` and rescaling with an upper bound :math:`R \geq \lVert B \rVert_2`,

        .. math::
            \exp(A)\,\psi = \exp(-iB)\,\psi
            = \sum_{k=0}^{K} (2 - \delta_{k,0})\,(-i)^k J_k(R)\,T_k(B/R)\,\psi,

    where :math:`J_k` are Bessel functions of the first kind. The Chebyshev polynomials are applied
    through the three term recurrence
    :math:`T_{k+1}(\xi)\psi = 2\xi\,T_k(\xi)\psi - T_{k-1}(\xi)\psi`, so that every term of the sum
    costs one product of the generator with the state.

    One propagation step therefore costs ``order`` matrix-vector products instead of one dense
    matrix exponential, which pays off when few states are propagated in a large Hilbert space.

    The expansion is truncated after ``order`` terms, which leaves an error of about
    :math:`2\lvert J_{K+1}(R)\rvert`. The Bessel coefficients fall off super-exponentially once
    :math:`k > R`, so the truncation is negligible as soon as the order exceeds the radius of a
    step, which is small for a resolution that resolves the dynamics. The radius is bounded by the
    largest row sum of the generator, a bound that is tight for the sparse Hamiltonians of most
    models but loose by up to :math:`\sqrt{\dim}` for dense ones. Since the truncation is not
    detected at runtime, use :meth:`suggested_order` to check that the order is large enough for
    the model at hand.

    Note:
        The radius is recomputed from the generators of every interval and does not carry a
        gradient, so that the gradient of ``_propagate``, and with it
        :class:`~paraqeet.propagation.auto_diff_gradients.AutoDiffGradients`, only differentiates
        the polynomial in the generator.

    Note:
        The expansion converges for every generator whose field of values is bounded by :math:`R`,
        which includes the generators of the Lindblad master equation and of the block
        triangular equation of motion of :class:`~paraqeet.propagation.goat.GOAT`. Those need an
        order slightly above the radius, while an anti-Hermitian generator converges already at
        :math:`K \approx R`.
    """

    _order: int

    def __init__(
        self,
        eom_func: Callable[[Array], Array],
        resolution: float,
        initial_state: Array,
        order: int = 16,
    ) -> None:
        """
        Args:
            eom_func: A function that gives the equation of motion.
            resolution: Propagation resolution used to solve the equation of motion.
                The corresponding time step dt = 1/resolution.
            initial_state: State at the beginning of the simulation.
            order: Number of Chebyshev terms of the expansion of the matrix exponential.
                Defaults to 16, which reaches machine precision for a step whose generator has a
                radius of about two, and an error of :math:`10^{-8}` at a radius of five.
                Use :meth:`suggested_order` to trade the margin of the default for speed.

        Raises:
            ConfigurationException: If the order is smaller than one.
        """
        super().__init__(eom_func, resolution, initial_state)
        if order < 1:
            raise ConfigurationException("The order of the Chebyshev expansion has to be at least one.")
        self._order = order

    @property
    def order(self) -> int:
        """Return the number of Chebyshev terms of the expansion."""
        return self._order

    @order.setter
    def order(self, order: int) -> None:
        """Set the number of Chebyshev terms of the expansion.

        Raises:
            ConfigurationException: If the order is smaller than one.
        """
        if order < 1:
            raise ConfigurationException("The order of the Chebyshev expansion has to be at least one.")
        self._order = order

    @staticmethod
    @jit
    def _radius(eom: Array) -> Array:
        """Return an upper bound on the two-norm of the generators of all steps.

        Bounds the largest two-norm by the geometric mean of the one-norm and the infinity-norm,
        both of which are cheap. For a Hermitian generator the two norms agree and the bound is
        tight. Vanishing generators are given a radius of one, for which the expansion returns the
        state unchanged.

        Args:
            eom: Equation of motion, already scaled with the step size, for a list of times.
        """
        row_sums = jnp.max(jnp.sum(jnp.abs(eom), axis=-1), axis=-1)
        column_sums = jnp.max(jnp.sum(jnp.abs(eom), axis=-2), axis=-1)
        radius = jnp.max(jnp.sqrt(row_sums * column_sums))
        return jnp.where(radius > 0.0, radius, 1.0)

    @staticmethod
    @partial(jit, static_argnums=(1,))
    def _coefficients(radius: Array, order: int) -> Array:
        r"""Return the expansion coefficients :math:`(2 - \delta_{k,0})\,(-i)^k J_k(R)`.

        The Bessel functions are summed from their power series

            .. math::
                J_k(R) = \sum_m \frac{(-1)^m}{m!\,(k + m)!}\left(\frac{R}{2}\right)^{k + 2m},

        evaluated through log-gamma functions to keep the terms in range. The series alternates,
        hence it loses accuracy for large radii, where the terms cancel to the order of
        :math:`e^{R}`. This is uncritical because an expansion of a practical order is restricted
        to small radii anyway.

        Args:
            radius: Radius the generators are rescaled with.
            order: Number of Chebyshev terms of the expansion.
        """
        num_terms = order + 40
        ks = jnp.arange(order + 1)[:, None]
        ms = jnp.arange(num_terms)[None, :]
        log_terms = (ks + 2 * ms) * jnp.log(radius / 2.0) - gammaln(ms + 1.0) - gammaln(ks + ms + 1.0)
        bessel = jnp.sum(jnp.where(ms % 2 == 0, 1.0, -1.0) * jnp.exp(log_terms), axis=1)

        prefactors = jnp.array([(2.0 - (k == 0)) * (-1j) ** k for k in range(order + 1)], dtype=jnp.complex128)
        return prefactors * bessel

    @override
    def _propagate_args(self, dt: Float) -> tuple[Array, ...]:
        """Return the indices of the terms of the expansion, one per Chebyshev polynomial.

        The order is handed to the propagation as the length of an array, the same way the number
        of propagation steps is, because it has to stay a static quantity of the compiled
        propagation while the arguments themselves are traced.

        Args:
            dt: Length of one propagation step. Unused, the expansion rescales itself.
        """
        return (jnp.arange(self._order + 1),)

    @staticmethod
    @jit
    @override
    def _propagate(eom: Array, psis_t: Array, steps_arr: Array, terms_arr: Array) -> Array:
        """Propagate the system in time with a truncated Chebyshev expansion.

        The iterations over the propagation steps use ``jax.lax.scan`` to avoid compilation
        overhead, while the terms of the expansion are unrolled because their number is static.

        Args:
            eom: Equation of motion, already scaled with the step size, for a list of times.
            psis_t: State/states at time 't'.
            steps_arr: Iteration indices, one per propagation step.
            terms_arr: Iteration indices, one per term of the expansion.

        Returns:
            The evolved state.
        """
        order = terms_arr.shape[0] - 1
        # The expansion is exact for every radius above the norm of the generator, hence no
        # gradient of the propagation flows through the rescaling.
        radius = stop_gradient(ExpmChebyshev._radius(eom))
        coefficients = stop_gradient(ExpmChebyshev._coefficients(radius, order))
        # Turns a product with the generator A = -iB into the argument B/R of the polynomials.
        scale = 1j / radius

        def propagate_body(psis_t: Array, index: Any) -> tuple[Array, Array]:
            generator = eom[index]
            chebyshev_prev = psis_t
            chebyshev = scale * (generator @ psis_t)
            propagated = coefficients[0] * chebyshev_prev + coefficients[1] * chebyshev
            for term in range(2, order + 1):
                chebyshev_prev, chebyshev = (
                    chebyshev,
                    2.0 * scale * (generator @ chebyshev) - chebyshev_prev,
                )
                propagated = propagated + coefficients[term] * chebyshev
            return propagated, propagated

        psis_t, _ = scan(propagate_body, psis_t, steps_arr)

        return psis_t

    def suggested_order(self, times: Array, tolerance: float = 1e-14, margin: int = 2) -> int:
        """Return the expansion order that the equation of motion needs on the given times.

        Samples the equation of motion the way :meth:`get_value` does, takes the largest radius of
        all steps and returns the first order whose Bessel coefficient drops below ``tolerance``.
        This is a diagnostic to run once for a model, and not in every optimization step, because
        it evaluates the equation of motion on the full time grid.

        Note:
            Wrapping the propagation in :class:`~paraqeet.propagation.goat.GOAT` propagates a block
            triangular generator that also contains the derivatives of the equation of motion. Its
            radius is larger than the one seen here, so add a few orders on top of the suggestion.

        Args:
            times: Array of times, as handed to :meth:`get_value`.
            tolerance: Largest acceptable Bessel coefficient of the first truncated term.
            margin: Number of terms added on top of the order that reaches the tolerance.

        Returns:
            The suggested value of ``order``.

        Raises:
            ValueError: If fewer than two time points are given.
        """
        if len(times) < 2:
            raise ValueError("ExpmChebyshev.suggested_order needs at least two time points.")

        radius = 0.0
        for ti in range(1, len(times)):
            step_times, dt = construct_times(times, ti, self._resolution)
            eom = self._eom_func(self._construct_time_grid(step_times, dt)) * dt
            radius = max(radius, float(ExpmChebyshev._radius(jnp.asarray(eom))))

        orders = np.arange(2 * int(radius) + 64)
        converged = np.nonzero(np.abs(jv(orders, radius)) < tolerance)[0]
        order = int(orders[converged[0]]) if converged.size > 0 else int(orders[-1])
        return max(order + margin, 1)
