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
from paraqeet.propagation.scipy_expm import ScipyExpm
from paraqeet.quantity import Array, Float

jax.config.update("jax_enable_x64", True)


class ScipyExpmGRAPE(ScipyExpm, Differentiable):
    """Solve EOMs by piecewise exponentiation via Scipy using GRAPE.

    Compute the gradients of a closed quantum system for PWC pulses by using
    GRAPE. Here, we use forward propagation of the initial state and backward
    propagation of the target state to compute the gradients.

    The `eom_func` function is required in addition to `eom_gradient_func` as a computationally
    "cheaper" alternative for cases where gradient information is not required, such as gradient-free optimization.

    The state propagations are done by the `ScipyExpm` method.

    _resolution: float
        Simulation resolution.
    _initial_state: Array = None
        Initial state for forward propagation.
    _target_state: Array = None
        Target state for backward propagation.
    _schirmer_derivative: bool = False
        If true, compute the gradient by Schirmer Derivative/Method of auxiliary
        matrix exponential. If false, use frechet derivative.
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
    ):
        ScipyExpm.__init__(self, eom_func, resolution, initial_state)
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

        Parameters
        ----------
        target_state: Array
            Target state.
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
    def operator_sandwich_function(self, operator_sandwich_func: Callable):
        """Set the step function for solving the backward propagation of the target state."""
        self._operator_sandwich_function = operator_sandwich_func

    @property
    def schirmer_derivative(self) -> bool:
        """Returns whether the Schirmer method is used to compute the derivative of the unitary operator."""
        return self._schirmer_derivative

    @schirmer_derivative.setter
    def schirmer_derivative(self, schirmer_derivative: bool) -> None:
        """Schirmer Derivative method to compute derivative of Unitary operator.

        Parameters
        ----------
        schirmer_derivative : bool
            If True use Schirmer derivative, if False use Frechet Derivative.
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

        Parameters
        ----------
        psis_t : Array
            Forward propagated state
        lamdas_t : Array
            Backward propagated state
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

        Parameters
        ----------
        psis_t: Array
            Forward propagated state
        lamdas_t: Array
            Backward propagated state
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
    def _exponentiate_frechet(dim, ham, dh_dp):
        r"""Exponentiate and also calculate the frechet derivative.

        Parameters
        ----------
        ham: Array
            -iHdt
        dh_dp: Array
            -i\frac{\partial H}{\partial u} dt
        """
        return expm_frechet(ham, dh_dp)

    @staticmethod
    @partial(jit, static_argnums=(0,))
    def _exponentiate_schirmer(dim, ham, dh_dp):
        r"""Exponentiate an auxiliary matrix to compute U and dU.

        Parameters
        ----------
        ham : Array
            -iHdt
        dh_dp : Array
            -i\frac{\partial H}{\partial u} dt
        """
        zeros = jnp.zeros_like(ham)
        h_extended = jnp.block([[ham, dh_dp], [zeros, ham]])
        u_extended = expm(h_extended)
        return u_extended[:dim, :dim], u_extended[:dim, dim:]

    @staticmethod
    @jit
    def _exponentiate(ham):
        r"""Exponentiate EOM using Expm.

        Parameters
        ----------
            ham : Array
                -iHdt
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

        Parameters
        ----------
        psis_t : Array
            Forward propagated state
        lamdas_t : Array
            Backward propagated state
        """

        def forward_propagation(psis_t, index):
            psis_t = us[index] @ psis_t
            return psis_t, psis_t

        psis_t, psis_list = scan(forward_propagation, psis_t, steps_arr)
        return psis_list

    @override
    def propagate(self, time: Array) -> Array:
        """Loop over all desired times in time at set resolution."""
        if len(time) < 2:
            raise ValueError("ScipyExpmGRAPE.propagate needs at least two time points.")

        if self._initial_state is None:
            raise ConfigurationException("Initial state is not set")

        init_state = jnp.array(self._initial_state, dtype=jnp.complex128)
        dt = time[1] - time[0]

        time_grid = time[:-1] + dt / 2

        eom = self._eom_func(time_grid) * dt

        us = vmap(ScipyExpmGRAPE._exponentiate, in_axes=(0,))(eom)

        psis = self._propagate_in_time(us, init_state, jnp.arange(0, len(time_grid), 1))
        psis = jnp.concat([jnp.expand_dims(init_state, axis=0), psis], axis=0)
        return jnp.array(psis)

    @override
    def get_value(self, times: Array) -> Float | Array:
        return self.propagate(times)

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
            exponentiating_function = ScipyExpmGRAPE._exponentiate_schirmer
        else:
            exponentiating_function = ScipyExpmGRAPE._exponentiate_frechet

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
