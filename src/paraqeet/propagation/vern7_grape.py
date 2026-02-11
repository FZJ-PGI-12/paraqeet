"""Class definition of the 7th-order Verner ODE solver for GRAPE."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import dynamic_slice_in_dim, scan

from paraqeet.exceptions import ConfigurationException
from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.propagation.differentiable_propagation import DifferentiablePropagation
from paraqeet.propagation.vern7 import Vern7
from paraqeet.quantity import Array

jax.config.update("jax_enable_x64", True)


class Vern7GRAPE(Vern7, DifferentiablePropagation):
    """
    Solve EOMs by 7th order ODE method to compute gradients using GRAPE.

    Compute the gradients of a quantum system for PWC pulses by using GRAPE.
    Here, we use forward propagation of the initial state and backward
    propagation of the target state to compute the gradients.

    The state propagations are done by the `Vern7 ODE` method.

    _resolution: float
        Simulation resolution.
    _initial_state: Array = None
        Initial state for forward propagation.
    _target_state: Array = None
        Target state for backward propagation.
    """

    _target_state: Array | None = None

    def __init__(self, model: EquationOfMotion, resolution: float):
        super().__init__(model, resolution)

        if self.is_open:
            self._reverse_step_function = self._reverse_lindblad_step
        else:
            self._reverse_step_function = self._reverse_schrodinger_step

    def set_target_state(self, target_state: Array) -> None:
        """Set target state for backward propagation.

        Parameters
        ----------
        target_state: Array
            Target state.
        """
        # For open system check if target state is a density matrixs.
        if self.is_open:
            if target_state.shape[-1] != target_state.shape[-2]:
                raise ConfigurationException(
                    f"Obtained a state vector of shape {target_state.shape} as target state. "
                    + "For open system propagation expected a density matrix as the target state."
                )

        self._target_state = target_state

    def _reverse_schrodinger_step(self, state: Array, h: Array, cols: list[Array]):
        return jnp.matmul(state, h)

    def _reverse_lindblad_step(self, state: Array, h: Array, cols: list[Array]):
        del_rho = Vern7._commutator(h, state)
        for col in cols:
            del_rho -= jnp.matmul(jnp.matmul(Vern7._dagger(col), state), col)
            del_rho += 0.5 * Vern7._anti_commutator(jnp.matmul(Vern7._dagger(col), col), state)
        return del_rho

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

        Parameters
        ----------
        psis_t: Array
            Forward propagated state
        lamdas_t: Array
            Backward propagated state
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

        if self._initial_state is None:
            raise ConfigurationException("Initial state is not set")

        if self._target_state is None:
            raise ConfigurationException("Target state is not set")

        init_state = jnp.array(self._initial_state, dtype=jnp.complex128)
        target_state = jnp.array(self._target_state, dtype=jnp.complex128)
        target_state = target_state.conj().T

        if self._model is None:
            raise ConfigurationException("No equation of motion is configured.")
        eom_func = self._model.get_value
        grad_func = self._model.get_value_and_gradient

        # Verify if `model.ode_propagation` is set to `True`.
        # ode_propgation returns hamiltonian and collapse operators separately.
        if not self._model.ode_propagation:
            raise ConfigurationException("Please set `model.ode_propagation` to `True` for this propagation method.")

        psis_list = [init_state]
        lamdas_list = [target_state]

        for ti in range(1, len(times)):
            psi_t = psis_list[ti - 1]
            lamda_t = lamdas_list[ti - 1]

            # Interpolate times
            time_grid, dt = self._construct_times(times, ti)
            times_interp = Vern7._interpolate_time(time_grid, dt)
            times_interp = times_interp + dt / 2

            if len(times_interp) < 9:
                raise ConfigurationException(
                    "Propagation resolution has been set very low. Higher resolution needed for this method."
                )

            # TODO: currently separate time grids are required for the EOM and the gradients.
            # TODO: Can we use one so that the value and gradients are computed simultaneously?

            eom, cols = eom_func(times_interp)

            psi_t, lamda_t = self._forward_and_backward_propagation(
                psi_t, lamda_t, eom * dt, jnp.array(cols) * jnp.sqrt(dt), jnp.arange(0, len(time_grid), 1)
            )

            psis_list.append(psi_t)
            lamdas_list.append(lamda_t)

        psis = jnp.array(psis_list)
        lamdas = jnp.array(lamdas_list)

        lamdas = jnp.flip(lamdas, axis=0)
        _, dh_dps = grad_func(times[:-1] + dt / 2)
        dh_dps = jnp.array(dh_dps) * dt

        grads = []
        n_params = dh_dps.shape[1]
        for i in range(n_params):
            if self.is_open:
                fwd_prop_state = vmap(Vern7._commutator, in_axes=(0, 0))(dh_dps[:, i, ...], psis[1:])
            else:
                fwd_prop_state = vmap(jnp.matmul, in_axes=(0, 0))(dh_dps[:, i, ...], psis[1:])

            grad = vmap(jnp.matmul, in_axes=(0, 0))(lamdas[1:], fwd_prop_state)
            grad = jnp.squeeze(grad)
            grads.append(grad)
        return psis, jnp.array(grads)
