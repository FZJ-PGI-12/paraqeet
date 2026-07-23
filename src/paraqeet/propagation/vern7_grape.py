"""Class definition of the 7th-order Verner ODE solver for GRAPE."""

from collections.abc import Callable
from functools import partial
from typing import override

import jax
import jax.numpy as jnp
from jax import jit
from jax.lax import dynamic_slice_in_dim, scan

from paraqeet.differentiable import Differentiable
from paraqeet.exceptions import ConfigurationException
from paraqeet.propagation.utils import construct_times
from paraqeet.propagation.vern7 import Vern7
from paraqeet.quantity import Array

jax.config.update("jax_enable_x64", True)


class Vern7GRAPE(Vern7, Differentiable):
    r"""
    Solve EOMs by 7th order ODE method and compute gradients using GRAPE.

    Compute the gradients of a quantum system for PWC pulses by using GRAPE.
    Here, we use forward propagation of the initial state and backward
    propagation of the target state to compute the gradients.

    The state propagations are done by the `Vern7 ODE` method.

    Attributes:
        _eom_and_gradient_func: Function that returns EOM and its gradient for an array of times.
        _target_state: Target state for backwards/reverse propagation for GRAPE.
        _operator_sandwich_function: Operator sandwich function to compute GRAPE gradients. It evaluates

            1. For closed system
                .. math::
                    \langle \lambda(t) \lvert \frac{\partial H}{\partial \alpha} \rvert \psi(t) \rangle

            2. For open system
                .. math::
                    \text{Tr}(\sigma(t) [H, \rho(t)])

        _reverse_step_function: Reverse step function for the backwards propagation.
    """

    _eom_gradient_func: Callable[[Array], Array]
    _target_state: Array
    _reverse_step_function: Callable
    _operator_sandwich_function: Callable

    def __init__(
        self,
        eom_func: Callable[[Array], Array],
        eom_gradient_func: Callable[[Array], Array],
        resolution: float,
        initial_state: Array,
        target_state: Array,
        step_function: Callable,
        reverse_step_function: Callable,
        operator_sandwich_function: Callable,
        jump_operators: list[Array] | None = None,
    ) -> None:
        r"""
        Args:
            eom_func: Equation of motion (EOM) as a function of time.
            eom_and_gradient_func: Function that returns EOM and its gradient for an array of times.
            resolution: Resolution at which to sample the EOM.
            initial_state: Initial state.
            target_state: Target state for backwards/reverse propagation for GRAPE.
            step_function: Step function used to that implements the right hand side of the EOM.
            jump_operators: A list of jump operators (each multiplied by the sqrt of the corresponding decay rate).
                Defaults to None for closed system.
            reverse_step_function: Reverse step function for the backwards propagation.
            operator_sandwich_function: Operator sandwich function to compute GRAPE gradients. It evaluates

                1. For closed system
                    .. math::
                        \langle \lambda(t) \lvert \frac{\partial H}{\partial \alpha} \rvert \psi(t) \rangle

                2. For open system
                    .. math::
                        \text{Tr}(\sigma(t) [H, \rho(t)])
        """
        super().__init__(eom_func, resolution, initial_state, step_function, jump_operators)
        self._eom_gradient_func = eom_gradient_func
        self._reverse_step_function = reverse_step_function
        self._target_state = target_state
        self._operator_sandwich_function = operator_sandwich_function

    @property
    def target_state(self) -> Array:
        """Return target state."""
        return self._target_state

    @target_state.setter
    def target_state(self, target_state: Array) -> None:
        """Set target state for backward propagation.

        Args:
            target_state: Target state.
        """
        # TODO: Provide explicit wrappers for multiple initial states or density vectors
        self._target_state = target_state

    @property
    def reverse_step_function(self) -> Callable:
        """Return the reverse step function for solving the backward propagation of the target state."""
        return self._reverse_step_function

    @reverse_step_function.setter
    def reverse_step_function(self, reverse_step_func: Callable) -> None:
        """Set the step function for solving the backward propagation of the target state."""
        self._reverse_step_function = reverse_step_func

    @property
    def operator_sandwich_function(self) -> Callable:
        r"""Return the operator sandwich function for computing the gradients.

        Closed system involves
            .. math::
                \langle \lambda(t) \lvert \frac{\partial H}{\partial \alpha} \rvert \psi(t) \rangle

        and open system involves
            .. math::
                \text{Tr}(\sigma(t) [H, \rho(t)])
        """
        return self._operator_sandwich_function

    @operator_sandwich_function.setter
    def operator_sandwich_function(self, operator_sandwich_func: Callable) -> None:
        """Set the step function for solving the backward propagation of the target state."""
        self._operator_sandwich_function = operator_sandwich_func

    @partial(jit, static_argnums=(0,))
    def _forward_and_backward_propagation(
        self,
        psis_t,
        lamdas_t,
        eom,
        col,
        steps_arr,
    ):
        """Forward propagate initial state and backward propagate target state.

        JIT compiled and uses `jax.lax.scan` to avoid compilation overhead.

        Args:
            psis_t: Forward propagated state
            lamdas_t: Backward propagated state
        """

        def forward_propagation(psis_t, index):
            psis_t = self._vern7_one_step(
                psis_t,
                dynamic_slice_in_dim(eom, start_index=9 * index, slice_size=9, axis=0),
                col,
            )
            return psis_t, psis_t

        def backward_propagation(lamdas_t, index):
            lamdas_t = self._vern7_one_step(
                lamdas_t,
                dynamic_slice_in_dim(eom, start_index=9 * index, slice_size=9, axis=0),
                col,
            )
            return lamdas_t, lamdas_t

        psis_t, _ = scan(forward_propagation, psis_t, steps_arr)

        self.step_function = self._reverse_step_function
        eom = (-1) * jnp.flip(eom, axis=0)
        lamdas_t, _ = scan(backward_propagation, lamdas_t, steps_arr)

        return psis_t, lamdas_t

    @override
    def get_gradient(self, times: Array) -> Array:
        _, gradient = self.get_value_and_gradient(times)
        return gradient

    @override
    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
        """Compute gradients using GRAPE.

        Compute the forward propagation of the initial state and
        the backward propagation of the target state.

        Psis represent the forward propagation and lamdas represent
        the backward propagation states.

        This propagation method assumes a PWC pulse as input.

        Note: This method only computes the first order gradients right now.
        """
        if len(times) < 2:
            raise ValueError("Vern7GRAPE.get_value_and_gradient needs at least two time points.")

        init_state = jnp.array(self._initial_state, dtype=jnp.complex128)
        target_state = jnp.array(self._target_state, dtype=jnp.complex128)
        target_state = target_state.conj().T

        psis_list = [init_state]
        lamdas_list = [target_state]

        for ti in range(1, len(times)):
            psi_t = psis_list[ti - 1]
            lamda_t = lamdas_list[ti - 1]

            # Interpolate times
            time_grid, dt = construct_times(times, ti, self._resolution)
            times_interp = Vern7._interpolate_time(time_grid, dt)
            times_interp = times_interp + dt / 2

            if len(times_interp) < 9:
                raise ConfigurationException(
                    "Propagation resolution has been set very low. Higher resolution needed for this method."
                )

            # TODO: currently separate time grids are required for the EOM and the gradients.
            # TODO: Can we use one so that the value and gradients are computed simultaneously?

            eom = self._eom_func(times_interp)

            psi_t, lamda_t = self._forward_and_backward_propagation(
                psi_t,
                lamda_t,
                eom * dt,
                self._jump_operators * jnp.sqrt(dt),
                jnp.arange(0, len(time_grid), 1),
            )

            psis_list.append(psi_t)
            lamdas_list.append(lamda_t)

        psis = jnp.array(psis_list)
        lamdas = jnp.array(lamdas_list)

        lamdas = jnp.flip(lamdas, axis=0)
        dh_dps = self._eom_gradient_func(times[:-1] + dt / 2)
        dh_dps = jnp.array(dh_dps) * dt

        grads = []
        n_params = dh_dps.shape[1]
        for i in range(n_params):
            grad = self._operator_sandwich_function(dh_dps[:, i, ...], psis[1:], lamdas[1:])
            grad = jnp.squeeze(grad)
            grads.append(grad)
        return psis, jnp.array(grads)
