"""Class definition of the 7th-order Verner ODE solver for GRAPE."""

from functools import partial
from jax import jit, vmap
from jax.lax import scan, dynamic_slice_in_dim
import jax.numpy as jnp

from paraqeet.quantity import Array
from paraqeet.exceptions import ConfigurationException
from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.propagation.vern7 import Vern7

import jax

jax.config.update("jax_enable_x64", True)


class Vern7GRAPE(Vern7):
    """
    Solve EOMs by 7th order ODE method to compute gradients using GRAPE.

    Compute the gradients of a quantum system for PWC pulses by using GRAPE.
    Here, we use forward propagation of the initial state and backward
    propagation of the target state to compute the gradients.

    The state propagations are done by the `Vern7 ODE` method.

    _res: float
        Simulation resolution.
    _initial_state: Array = None
        Initial state for forward propagation.
    _target_state: Array = None
        Target state for backward propagation.
    """

    _target_state: Array | None = None

    def __init__(self, model: EquationOfMotion, res: float):
        super().__init__(model, res)

        if self.is_open:
            self.__reverse_step_function = self._reverse_lindblad_step
        else:
            self.__reverse_step_function = self._reverse_schrodinger_step

    @property
    def target_state(self) -> Array | None:
        """Returns the current target state for backward propagation."""
        return self._target_state

    @target_state.setter
    def target_state(self, targetState: Array) -> None:
        """Set target state for backward propagation.

        Parameters
        ----------
        targetState : Array
            Target state.
        """
        # For open system check if target state is a density matrixs.
        if self.is_open:
            if targetState.shape[-1] != targetState.shape[-2]:
                raise ConfigurationException(
                    f"Obtained a state vector of shape {targetState.shape} as target state. "
                    + "For open system propagation expected a density matrix as the target state."
                )

        self._target_state = targetState

    def _reverse_schrodinger_step(self, state: Array, h: Array, cols: list[Array]):
        return jnp.matmul(state, h)

    def _reverse_lindblad_step(self, state: Array, h: Array, cols: list[Array]):
        del_rho = self._commutator(h, state)
        for col in cols:
            del_rho -= jnp.matmul(jnp.matmul(self._dagger(col), state), col)
            del_rho += 0.5 * self._anti_commutator(jnp.matmul(self._dagger(col), col), state)
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
        """Forward propagate inital state and backward propagate target state.

        JIT compiled and uses `jax.lax.scan` to avoid compilation overhead.

        Parameters
        ----------
        psis_t : Array
            Forward propagated state
        lamdas_t : Array
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

        psis_t, psis_list = scan(forward_propagation, psis_t, steps_arr)

        self.step_function = self.__reverse_step_function
        eom = (-1) * jnp.flip(eom, axis=0)
        lamdas_t, lamdas_list = scan(backward_propagation, lamdas_t, steps_arr)

        return psis_list, lamdas_list

    def gradient(self, time: Array) -> tuple[Array, Array]:
        """Compute gradients using GRAPE.

        Compute the forward propagation of the initial state and
        the backward propagation of the target state.

        Psis represent the forward propagation and lamdas represent
        the backward propagation states.

        This propagation method assumes a PWC pulse as input.

        Note: This method only computes the first order gradients right now.
        """
        if self._initial_state is None:
            raise ConfigurationException("Initial state is not set")

        if self._target_state is None:
            raise ConfigurationException("Target state is not set")

        init_state = jnp.array(self._initial_state, dtype=jnp.complex128)
        target_state = jnp.array(self._target_state, dtype=jnp.complex128)
        target_state = target_state.conj().T

        if self._model is None:
            raise ConfigurationException("No equation of motion is configured.")
        eom_func = self._model.get_matrix
        grad_func = self._model.gradient

        # Verify if `model.ode_propagation` is set to `True`.
        # ode_propgation returns hamiltonian and collapse operators separately.
        eom_parts = eom_func(jnp.array([0]))
        if len(eom_parts) != 2:
            raise ConfigurationException("Please set `model.ode_propagation` to `True` for this propagation method.")

        dt = time[1] - time[0]
        interp_time = self._interpolate_time(time, dt)
        timeGrid = interp_time[:-1] + dt / 2

        eom, cols = eom_func(timeGrid)
        dH_dps = jnp.array(grad_func(time[:-1] + dt / 2)) * dt

        psis, lamdas = self._forward_and_backward_propagation(
            init_state, target_state, eom * dt, jnp.array(cols) * jnp.sqrt(dt), jnp.arange(0, len(time[:-1]), 1)
        )

        psis = jnp.concat([jnp.expand_dims(init_state, axis=0), psis], axis=0)
        lamdas = jnp.concat([jnp.expand_dims(target_state, axis=0), lamdas], axis=0)

        lamdas = jnp.flip(lamdas, axis=0)

        grads = []

        n_params = dH_dps.shape[1]
        for i in range(n_params):
            if self.is_open:
                fwd_prop_state = vmap(self._commutator, in_axes=(0, 0))(dH_dps[:, i, ...], psis[1:])
            else:
                fwd_prop_state = vmap(jnp.matmul, in_axes=(0, 0))(dH_dps[:, i, ...], psis[1:])

            grad = vmap(jnp.matmul, in_axes=(0, 0))(lamdas[1:], fwd_prop_state)
            grad = jnp.squeeze(grad)
            grads.append(grad)
        return psis, jnp.array(grads)
