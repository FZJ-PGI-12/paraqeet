"""Class definition of the JAX piecewise exponential propagation model."""

from typing import Any

import jax
from jax import jit
from jax.lax import scan
from jax.scipy.linalg import expm

from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array

jax.config.update("jax_enable_x64", True)


class Expm(Propagation):
    """Piecewise matrix exponential propagation system.

    Solve the equation of motion by piecewise exponentiation with the
    JAX package.

    """

    @staticmethod
    @jit
    def _propagate(eom: Array, psis_t: Array, steps_arr: Array) -> Array:
        """Propagate the system in time.

        Iteratively propagate state/states (psis_t) according
        to the equation of motion (eom). The eom is exponentiated using
        ``jax.scipy.linalg.expm`` to compute the propagators.
        The iterations use ``jax.lax.scan`` to avoid compilation overhead.

        Args:
            psis_t: State/states at time 't'.
            eom: Equation of motion for a list of times.
            steps_arr: Array from 0 to the length of the list of times, in steps
                of 1 representing the iteration index.

        Returns:
            The evolved state.

        """

        def propagate_body(psis_t: Array, index: Any) -> tuple[Array, Array]:
            psis_t = Expm._propagate_psi(eom[index], psis_t)
            return psis_t, psis_t

        psis_t, _ = scan(propagate_body, psis_t, steps_arr)

        return psis_t

    @staticmethod
    @jit
    def _propagate_psi(eom_matrix: Array, psis_t: Array) -> Array:
        """Propagate the state/states (psis_t).

        Args:
            eom_matrix: The equations of motion matrix.
            psis_t: State/states at time 't'.

        Returns:
            Array: Returns the evolved state.

        """
        propagated: Array = expm(eom_matrix) @ psis_t
        return propagated
