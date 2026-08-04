"""Compute gradient of a propagation method by automatic differentiation."""

from collections.abc import Callable
from typing import override

import jax.numpy as jnp
from jax import jit

from paraqeet.autograd_utils import get_value_and_jacobian_rev
from paraqeet.propagation.propagation import DifferentiablePropagation, Propagation
from paraqeet.propagation.utils import construct_times
from paraqeet.quantity import Array


class AutoDiffGradients(DifferentiablePropagation):
    r"""Wraps a Propagation object to add gradient computation using Automatic differentiation.

    The class provides gradients of the propagation method using automatic differentiation (AD)
    assuming a functionally pure ``_propagate()`` method is defined that is compatible with
    JAX JIT and vjp (reverse-mode AD).

    The AD performs
        .. math::
            \frac{\partial \ket{\psi(t)}}{\partial \alpha} = \sum_{\tau \in [0, t]}
            \frac{\partial \ket{\psi(t)}}{\partial H(\tau)} \frac{\partial H(\tau)}{\partial \alpha}
            + \sum_{\tau \in [0, t_1, t_2, ...]} \frac{\partial \ket{\psi(t)}}{\partial \ket{\psi(\tau)}}
            \frac{\partial \ket{\psi(\tau)}}{\partial \alpha}


    where :math:`\alpha` is some pulse parameter and the second term on the right hand side is due to checkpointing.
    The term :math:`\frac{\partial \ket{\psi(t)}}{\partial H(\tau)}` can be computed by AD of the propagation method,
    and :math:`\frac{\partial H(\tau)}{\partial \alpha}` is provided by the user as `eom_gradient_func`.
    """

    _propagation_and_gradient_func: Callable[..., tuple[Array, Array]]

    def __init__(
        self,
        propagation: Propagation,
        eom_gradient_func: Callable[[Array], Array],
    ):
        """
        Args:
            propagation: A JAX jit and vjp compatible propagation object.
            eom_gradient_func: Function that returns the gradient of EOM.
        """
        super().__init__(propagation, eom_gradient_func)
        self._propagation_and_gradient_func = jit(get_value_and_jacobian_rev(self._prop._propagate, argnums=(0, 1)))

    @override
    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
        """Return value and gradient of propagation.

        Default implementation utilizing automatic differentiation of ``_propagate``
        method using ``jax.vjp``. Overwrite this method to implement your own
        ``get_value_and_gradient`` method.

        Args:
            times: Array of time points/
        """
        psis = [self._prop.initial_state]
        # The state at the initial time does not depend on the parameters.
        grads = [jnp.zeros((self._eom_gradient_func(jnp.array([0.0])).shape[1],) + psis[0].shape, dtype=psis[0].dtype)]

        for ti in range(1, len(times)):
            step_times, dt = construct_times(times, ti, self._prop.resolution)
            psis_t = psis[ti - 1]
            eom, eom_grads_t = self._eom_and_gradient_func(self._prop._construct_time_grid(step_times, dt))
            psis_t, (eom_jacobian, state_jacobian) = self._propagation_and_gradient_func(
                eom * dt,
                psis_t,
                jnp.arange(0, len(step_times), 1),
                *self._prop._propagate_args(dt),
            )

            # Contribution of the EOM of this interval to the gradient.
            eom_jacobian = jnp.transpose(eom_jacobian, axes=(2, 0, 1, 3, 4))
            grads_t = jnp.einsum("tnmjk, tpjk -> pnm", eom_jacobian, eom_grads_t) * dt
            # Chain rule through the state, which carries the gradient of all earlier intervals.
            grads_t += jnp.einsum("nmjk, pjk -> pnm", state_jacobian, grads[ti - 1])

            psis.append(psis_t)
            grads.append(grads_t)

        return jnp.array(psis), jnp.array(grads)

    @override
    def get_gradient(self, times: Array) -> Array:
        """Return gradient of propagated state/propagator w.r.t. specified parameters.

        Args:
            times: Array of time steps.
        """
        _, grads = self.get_value_and_gradient(times)
        return grads
