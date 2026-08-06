"""Compute gradients of propagation method by using the GOAT method."""

from collections.abc import Callable
from typing import override

import jax.numpy as jnp
from jax import jit, vmap

from paraqeet.propagation.propagation import DifferentiablePropagation, Propagation
from paraqeet.propagation.utils import construct_times
from paraqeet.quantity import Array


class GOAT(DifferentiablePropagation):
    r"""Compute gradient of any propagation using GOAT :cite:p:`machnes2018tunable`.

    GOAT solves the equation of motion of the state together with the equations of motion of
    its derivatives, by propagating a "super state"

        .. math::
            \Psi = (\psi, \partial_1 \psi, \dots, \partial_P \psi)

    under the lower triangular equation of motion

        .. math::
            \begin{pmatrix}
                A            & 0      & \cdots & 0      \\
                \partial_1 A & A      &        & 0      \\
                \vdots       &        & \ddots &        \\
                \partial_P A & 0      & \cdots & A
            \end{pmatrix}
    
    This extended EOM is solved by the ``_propagate`` method of
    *any* :class:`~paraqeet.propagation.propagation.Propagation` method.
    """

    def __init__(
        self,
        propagation: Propagation,
        eom_gradient_func: Callable[[Array], Array],
    ):
        """
        Args:
            propagation: Any propagation object.
            eom_gradient_func: Function that returns the gradient of EOM.
        """
        super().__init__(propagation, eom_gradient_func)

    @staticmethod
    @jit
    def _create_goat_eom(eom: Array, eom_grads: Array) -> Array:
        """Create the GOAT lower triangular EOM matrix.

        Builds the lower triangular matrix for all time points at once.

        Args:
            eom: Equation of motion with shape ``(n_times, dim, dim)``.
            eom_grads: Gradient of the EOM with shape ``(n_times, n_params, dim, dim)``.

        Returns:
            Array of shape ``(n_times, (n_params + 1) * dim, (n_params + 1) * dim)``.
        """
        n_times, n_params = eom_grads.shape[0], eom_grads.shape[1]
        dim = eom.shape[-1]
        size = (n_params + 1) * dim

        goat_eom = jnp.zeros((n_times, size, size), dtype=eom.dtype)
        for ii in range(n_params + 1):
            goat_eom = goat_eom.at[:, ii * dim : (ii + 1) * dim, ii * dim : (ii + 1) * dim].set(eom)
        return goat_eom.at[:, dim:, :dim].set(eom_grads.reshape(n_times, n_params * dim, dim))

    @staticmethod
    def _create_super_state(psi: Array, dpsis: Array) -> Array:
        """Create a super state from the state of the system and its derivatives.

        Args:
            psi: State of the system with shape ``(dim, n_states)``.
            dpsis: Derivatives of the state with shape ``(n_params, dim, n_states)``.

        Returns:
            Array: The super state with shape ``((n_params + 1) * dim, n_states)``.

        """
        return jnp.concatenate([psi, *dpsis], axis=0)

    @staticmethod
    def _decompose_super_state(super_state: Array, n_params: int, dim: int) -> tuple[Array, Array]:
        """Split a super state back into the state of the system and its derivatives.

        Args:
            super_state: Super state as returned by the propagation.
            n_params: Number of parameters.
            dim: Dimension of the system.

        Returns:
            A tuple ``(psi, dpsis)``.

        """
        psi = super_state[:dim]
        dpsis = super_state[dim:].reshape((n_params, dim) + super_state.shape[1:])
        return psi, dpsis

    @override
    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
        """Solve the GOAT equation for the gradient vector.

        If ``self._prop.batched_propagation`` is ``True`` the propagation and its gradient
        is computed in one compiled loop. *Input ``times`` has to be uniformly spaced in this case.*

        Args:
            times: Array of times.

        """
        if len(times) < 2:
            raise ValueError("GOAT.get_value_and_gradient needs at least two time points.")

        n_params = self._eom_gradient_func(jnp.array([0.0])).shape[1]
        dim = self._prop.initial_state.shape[0]

        if self._prop.batched_propagation:
            initial_super_state = GOAT._create_super_state(
                jnp.array(self._prop.initial_state, dtype=jnp.complex128),
                jnp.zeros((n_params,) + self._prop.initial_state.shape, dtype=jnp.complex128),
            )

            sampled_eom_and_gradient = self._sample_eom_and_gradient_batched(times)
            if sampled_eom_and_gradient is not None:
                eom, eom_grads, dt, steps = sampled_eom_and_gradient
                n_segments, n_samples = eom.shape[:2]

                goat_eom = GOAT._create_goat_eom(
                    jnp.reshape(eom, (n_segments * n_samples,) + eom.shape[2:]),
                    jnp.reshape(eom_grads, (n_segments * n_samples,) + eom_grads.shape[2:]),
                )
                goat_eom = jnp.reshape(goat_eom, (n_segments, n_samples) + goat_eom.shape[1:])

                super_states = self._prop._propagate_batched(
                    goat_eom,
                    initial_super_state,
                    steps,
                    *self._prop._propagate_args(dt),
                )
                psis_and_dpsis: tuple[Array, Array] = vmap(GOAT._decompose_super_state, in_axes=(0, None, None))(
                    super_states, n_params, dim
                )
                return psis_and_dpsis

        # Fallback to python loop

        psis: list[Array] = [jnp.array(self._prop.initial_state, dtype=jnp.complex128)]
        dpsis: list[Array] = [jnp.zeros((n_params,) + self._prop.initial_state.shape, dtype=jnp.complex128)]

        for ti in range(1, len(times)):
            step_times, dt = construct_times(times, ti, self._prop.resolution)
            sample_times = self._prop._construct_time_grid(step_times, dt)

            eom, eom_grads = self._eom_and_gradient_func(sample_times)
            goat_eom = GOAT._create_goat_eom(jnp.array(eom) * dt, jnp.array(eom_grads) * dt)

            super_state = self._prop._propagate(
                goat_eom,
                GOAT._create_super_state(psis[-1], dpsis[-1]),
                jnp.arange(0, len(step_times), 1),
                *self._prop._propagate_args(dt),
            )

            psi, dpsi = GOAT._decompose_super_state(super_state, n_params, dim)
            psis.append(psi)
            dpsis.append(dpsi)

        return jnp.array(psis), jnp.array(dpsis)

    @override
    def get_gradient(self, times: Array) -> Array:
        """Return the gradient of the propagated state/propagator w.r.t. the specified parameters.

        Args:
            times: Array of times.
        """
        _, gradient = self.get_value_and_gradient(times)
        return gradient
