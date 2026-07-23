"""Class definition of the Scipy piecewise exponential propagation model.

Uses the GRAPE optimization method.
Assumes that the signal is piecewise constant (PWC) without an LO and the
Hamiltonian is defined in the rotating frame of drive.

"""

from collections.abc import Callable
from functools import partial
from typing import override

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan
from jax.scipy.linalg import expm, expm_frechet

from paraqeet.differentiable import Differentiable
from paraqeet.exceptions import ConfigurationException
from paraqeet.propagation.expm import Expm
from paraqeet.quantity import Array

jax.config.update("jax_enable_x64", True)


class ExpmGRAPE(Expm, Differentiable):
    """Solve EOMs by piecewise exponentiation via Scipy using GRAPE.

    Compute the gradients of a closed quantum system for PWC pulses by using
    GRAPE. Here, we use forward propagation of the initial state and backward
    propagation of the target state to compute the gradients.

    The state propagations are done by the `ScipyExpm` method.

    Attributes:
        _resolution (float): Simulation resolution.
        _initial_state (Array, optional): Initial state for forward propagation. Defaults to None.
        _target_state (Array, optional): Target state for backward propagation. Defaults to None.
        _schirmer_derivative (bool): If true, compute the gradient by Schirmer Derivative/Method of auxiliary
            matrix exponential. If false, use frechet derivative. Defaults to False.
    """

    _eom_gradient_func: Callable[[Array], Array]
    _target_state: Array
    _schirmer_derivative: bool = False
    _operator_sandwich_function: Callable

    def __init__(
        self,
        eom_func: Callable[[Array], Array],
        eom_gradient_func: Callable[[Array], Array],
        resolution: float,
        initial_state: Array,
        target_state: Array,
        operator_sandwich_function: Callable,
    ) -> None:
        """
        Args:
            eom_func: A function that gives the equation of motion.
            eom_gradient_func: A function that gives the gradient of the equation of motion.
            resolution: Propagation resolution used to solve the equation of motion.
                The corresponding time step dt = 1/resolution.
            initial_state: State at the beginning of the simulation.
            target_state: Target state at the end of the simulation.
            operator_sandwich_function: Function for backpropagating the target state.
        """
        super().__init__(eom_func, resolution, initial_state)
        self._eom_gradient_func = eom_gradient_func
        self.target_state = target_state
        self._operator_sandwich_function = operator_sandwich_function

    @property
    def target_state(self):
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
    def operator_sandwich_function(self):
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

    @property
    def schirmer_derivative(self) -> bool:
        """Returns whether the Schirmer method is used to compute the derivative of the unitary operator."""
        return self._schirmer_derivative

    @schirmer_derivative.setter
    def schirmer_derivative(self, schirmer_derivative: bool) -> None:
        """Schirmer Derivative method to compute derivative of Unitary operator.

        Args:
            schirmer_derivative: If True use Schirmer derivative, if False use Frechet Derivative.
        """
        self._schirmer_derivative = schirmer_derivative

    @partial(jit, static_argnums=(0,))
    def _forward_and_backward_propagation(
        self,
        us,
        psis_t,
        lamdas_t,
        steps_arr,
    ):
        """Forward propagate initial state and backward propagate target state.

        JIT compiled and uses `jax.lax.scan` to avoid compilation overhead.

        Args:
            us: Unitaries at different times.
            psis_t: Forward propagated state
            lamdas_t: Backward propagated state
                of 1 representing the iteration index.
            steps_arr: Array from 0 to the length of the List of times, in steps
        """

        def forward_propagation(psis_t, index):
            psis_t = us[index] @ psis_t
            return psis_t, psis_t

        def backward_propagation(lamdas_t, index):
            lamdas_t = lamdas_t @ us[-index - 1]
            return lamdas_t, lamdas_t

        psis_t, psis_list = scan(forward_propagation, psis_t, steps_arr)
        lamdas_t, lamdas_list = scan(backward_propagation, lamdas_t, steps_arr)

        return psis_list, lamdas_list

    @partial(jit, static_argnums=(0,))
    def _forward_and_backward_propagation_open(
        self,
        us,
        us_rev,
        psis_t,
        lamdas_t,
        steps_arr,
    ):
        """Forward propagate initial state and backward propagate target state.

        JIT compiled and uses `jax.lax.scan` to avoid compilation overhead.

        Args:

            us: Unitaries at different times.
            us_rev: Inverse of the unitaries at different times.
            psis_t: Forward propagated state.
            lamdas_t: Backward propagated state.
            steps_arr: Array from 0 to the length of the List of times, in steps
                of 1 representing the iteration index.
        """

        def forward_propagation(psis_t, index):
            psis_t = us[index] @ psis_t
            return psis_t, psis_t

        def backward_propagation(lamdas_t, index):
            lamdas_t = us_rev[index] @ lamdas_t
            return lamdas_t, lamdas_t

        psis_t, psis_list = scan(forward_propagation, psis_t, steps_arr)
        lamdas_t, lamdas_list = scan(backward_propagation, lamdas_t, steps_arr)

        return psis_list, lamdas_list

    @staticmethod
    @partial(jit, static_argnums=(0,))
    def _exponentiate_frechet(dim: int, ham: Array, dh_dp: Array):
        r"""Exponentiate and also calculate the frechet derivative.

        Args:
            ham: -iHdt
            dh_dp: -i\frac{\partial H}{\partial u} dt
        """
        return expm_frechet(ham, dh_dp)

    @staticmethod
    @partial(jit, static_argnums=(0,))
    def _exponentiate_schirmer(dim, ham: Array, dh_dp: Array):
        r"""Exponentiate an auxiliary matrix to compute U and dU.

        Args:
            ham: -iHdt
            dh_dp: -i\frac{\partial H}{\partial u} dt
        """
        zeros = jnp.zeros_like(ham)
        h_extended = jnp.block([[ham, dh_dp], [zeros, ham]])
        u_extended = expm(h_extended)
        return u_extended[:dim, :dim], u_extended[:dim, dim:]

    @staticmethod
    @jit
    def _exponentiate(ham):
        r"""Exponentiate EOM using Expm.

        Args:
            ham: -i H dt.

        Returns:
            The expontial of -i H dt.
        """
        return expm(ham)

    @partial(jit, static_argnums=(0,))
    def _propagate_in_time(
        self,
        us,
        psis_t,
        steps_arr,
    ):
        """Propagate Full time.

        JIT compiled and uses `jax.lax.scan` to avoid compilation overhead.

        Args:
            us: Unitaries at different times.
            psis_t: Forward propagated state
            steps_arr: Array from 0 to the length of the List of times, in steps
        """

        def forward_propagation(psis_t, index):
            psis_t = us[index] @ psis_t
            return psis_t, psis_t

        psis_t, psis_list = scan(forward_propagation, psis_t, steps_arr)
        return psis_list

    @override
    def get_value(self, times: Array) -> Array:
        """Loop over all desired times in time at set resolution."""
        if len(times) < 2:
            raise ValueError("ScipyExpmGRAPE.propagate needs at least two time points.")

        if self._initial_state is None:
            raise ConfigurationException("Initial state is not set")

        init_state = jnp.array(self._initial_state, dtype=jnp.complex128)
        dt = times[1] - times[0]

        time_grid = times[:-1] + dt / 2

        eom = self._eom_func(time_grid) * dt

        us = vmap(ExpmGRAPE._exponentiate, in_axes=(0,))(eom)

        psis = self._propagate_in_time(us, init_state, jnp.arange(0, len(time_grid), 1))
        psis = jnp.concat([jnp.expand_dims(init_state, axis=0), psis], axis=0)
        return jnp.array(psis)

    @override
    def get_gradient(self, times: Array) -> Array:
        _, gradient = self.get_value_and_gradient(times)
        return gradient

    @override
    def get_value_and_gradient(self, times: Array) -> tuple:
        """Compute gradients using GRAPE.

        Compute the forward propagation of the initial state and
        the backward propagation of the target state.

        Psis represent the forward propagation and lamdas represent
        the backward propagation states.

        This propagation method assumes a PWC pulse as input.
        """
        # TODO: Test if this get_value_and_gradient method also works for open systems.

        if len(times) < 2:
            raise ValueError("ScipyExpmGRAPE.get_value_and_gradient needs at least two time points.")

        init_state = jnp.array(self._initial_state, dtype=jnp.complex128)
        target_state = jnp.array(self._target_state, dtype=jnp.complex128)
        target_state = target_state.conj().T

        dt = times[1] - times[0]
        time_grid = times[:-1] + dt / 2

        hams = self._eom_func(time_grid)
        dh_dps = self._eom_gradient_func(time_grid)
        hams = hams * dt
        dh_dps = jnp.array(dh_dps) * dt

        u_grads_list = []
        n_params = dh_dps.shape[1]

        dim = hams.shape[-2]

        if self._schirmer_derivative:
            exponentiating_function = ExpmGRAPE._exponentiate_schirmer
        else:
            exponentiating_function = ExpmGRAPE._exponentiate_frechet

        for i in range(n_params):
            us, d_us = vmap(exponentiating_function, in_axes=(None, 0, 0))(dim, hams, dh_dps[:, i, ...])
            u_grads_list.append(d_us)

        u_grads = jnp.stack(u_grads_list, axis=1)

        psis, lamdas = self._forward_and_backward_propagation(
            us, init_state, target_state, jnp.arange(0, len(time_grid), 1)
        )

        psis = jnp.concat([jnp.expand_dims(init_state, axis=0), psis], axis=0)
        lamdas = jnp.concat([jnp.expand_dims(target_state, axis=0), lamdas], axis=0)

        lamdas = jnp.flip(lamdas, axis=0)

        grads = []
        for i in range(n_params):
            grad = self._operator_sandwich_function(u_grads[:, i, ...], psis[:-1], lamdas[1:])
            grad = jnp.squeeze(grad)
            grads.append(grad)

        return psis, jnp.array(grads)
