"""Class definition of the Euler propagation model."""

from typing import Any, override

from jax import jit
from jax.lax import scan

from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array


class Euler(Propagation):
    r"""Simple implementation of first order Euler propagation.

    Solves the equation of motion d/dt psi(t) = F(psi(t), t)
    with a finite step size d as psi(t+d) = psi(t) + F(psi(t), t).
    The step size is determined by the propagation ``resolution``.
    """

    @staticmethod
    @jit
    @override
    def _propagate(eom: Array, psis_t: Array, steps_arr: Array) -> Array:
        """Propagate the system in time with first order Euler steps.

        Args:
            eom: Equation of motion, already scaled with the step size, for a list of times.
            psis_t: State/states at time 't'.
            steps_arr: Iteration indices, one per propagation step.

        Returns:
            The evolved state.

        """

        def propagate_body(psis_t: Array, index: Any) -> tuple[Array, Array]:
            psis_t = psis_t + eom[index] @ psis_t
            return psis_t, psis_t

        psis_t, _ = scan(propagate_body, psis_t, steps_arr)

        return psis_t
