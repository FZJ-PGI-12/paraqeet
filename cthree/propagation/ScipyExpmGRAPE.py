"""Class definition of the Scipy piecewise exponential propagation model.

Uses the GRAPE optimisation method.
Assumes that the signal is piecewise constant (PWC) without an LO and the
Hamiltonian is defined in the rotating frame of drive.

"""

from functools import partial

import numpy as np
import jax.numpy as jnp

from jax import jit, vmap
from jax.lax import scan

from jax.scipy.linalg import expm, expm_frechet

from cthree.model.Model import Model
from cthree.Exceptions import ConfigurationException
from cthree.propagation.ScipyExpm import ScipyExpm

import jax

jax.config.update("jax_enable_x64", True)


class ScipyExpmGRAPE(ScipyExpm):
    """Solve EOMs by piecewise exponentation via Scipy using GRAPE.

    Compute the gradients of a closed quantum system for PWC pulses by using
    GRAPE. Here, we use forward propagation of the initial state and backward
    propagation of the target state to compute the gradients.

    The state propagations are done by the `ScipyExpm` method.

    _res: float
        Simulation resolution.
    _initialState: np.ndarray = None
        Initial state for forward propagation.
    _targetState: np.ndarray = None
        Target state for backward propagation.
    _saveBwdPropagatedStates: bool = False
        Flag for saving backward propgated state. Saved if True.
    _bwdPropagatedStates: np.ndarray = None
        If `_saveBwdPropagatedStates` is True, save the bwd propagated states.
    _schirmerDerivative: bool = False
        If true, compute the gradient by Schirmer Derivative/Method of auxillary
        matrix exponential. If false, use frechet derivative.
    """

    _res: float
    _initialState: np.ndarray = None
    _targetState: np.ndarray = None
    _saveBwdPropagatedStates: bool = False
    _bwdPropagatedStates: np.ndarray = None
    _schirmerDerivative: bool = False

    def __init__(self, model: Model, res: float):
        super().__init__(model, res)

    def setTargetState(self, targetState: np.ndarray):
        """Set target state for backward propagation.

        Parameters
        ----------
        targetState : np.ndarray
            Target state.
        """
        self._targetState = targetState

    def setSaveBwdPropagatedStates(self, saveBwdPropagatedStates: bool):
        """Flag to save backwards propagated target state result.

        Parameters
        ----------
        saveBwdPropagatedStates : bool
            Save the states if True.
        """
        self._saveBwdPropagatedStates = saveBwdPropagatedStates

    def setSchirmerDerivative(self, schirmerDerivative: bool):
        """Schirmer Derivative method to compute derivative of Unitary operator.

        Parameters
        ----------
        schirmerDerivative : bool
            If True use Schirmer derivative, if False use Frechet Derivative.
        """
        self._schirmerDerivative = schirmerDerivative

    @staticmethod
    @jit
    def __sandwichOpValues(
        bwd_propagated_state: np.ndarray,
        Op: np.ndarray,
        fwd_propagated_state: np.ndarray,
    ) -> np.ndarray:
        r"""Compute \\langle \\lambda(t) | O | \\psi(t) \\rangle.

        Parameters
        ----------
        bwd_propagated_state : np.ndarray
            Backwards propagated states
        Op : np.ndarray
            Array of operator for each time point.
        fwd_propagated_state : np.ndarray
            Forwards propagated states

        Returns
        -------
        np.ndarray
            Matrix element of the operator for each time point.
        """
        return jnp.matmul(
            bwd_propagated_state, jnp.matmul(Op, fwd_propagated_state)
        )

    @partial(jit, static_argnums=(0,))
    def _ForwardAndBackwardPropagation(
        self,
        Us,
        psis_t,
        lamdas_t,
        steps_arr,
    ):
        """Forward propagate inital state and backward propagate target state.

        JIT compiled and uses `jax.lax.scan` to avoid compilation overhead.

        Parameters
        ----------
        psis_t : np.ndarray
            Forward propagated state
        lamdas_t : np.ndarray
            Backward propagated state
        """

        def ForwardPropagation(psis_t, index):
            psis_t = Us[index] @ psis_t
            return psis_t, psis_t

        def BackwardPropagation(lamdas_t, index):
            lamdas_t = lamdas_t @ Us[index]
            return lamdas_t, lamdas_t

        psis_t, psis_list = scan(ForwardPropagation, psis_t, steps_arr)
        lamdas_t, lamdas_list = scan(BackwardPropagation, lamdas_t, steps_arr)

        return psis_list, lamdas_list

    @staticmethod
    @partial(jit, static_argnums=(0,))
    def _exponentiateFrechet(dim, ham, dh_dp):
        r"""Exponentiate and also calculate the frechet derivative.

        Parameters
        ----------
        ham : np.ndarray
            -iHdt
        dh_dp : np.ndarray
            -i\\frac{\\partial H}{\\partial u} dt
        """
        return expm_frechet(ham, dh_dp)

    @staticmethod
    @partial(jit, static_argnums=(0,))
    def _exponentiateSchirmer(dim, ham, dh_dp):
        r"""Exponentiate an auxilliary matrix to compute U and dU.

        Parameters
        ----------
        ham : np.ndarray
            -iHdt
        dh_dp : np.ndarray
            -i\\frac{\\partial H}{\\partial u} dt
        """
        zeros = jnp.zeros_like(ham)
        H_extended = jnp.block([[ham, dh_dp], [zeros, ham]])
        U_extended = expm(H_extended)
        return U_extended[:dim, :dim], U_extended[:dim, dim:]

    @staticmethod
    @jit
    def _exponentiate(ham):
        r"""Exponentiate EOM using Expm.

        Parameters
        ----------
            ham : np.ndarray
            -iHdt
        """
        return expm(ham)

    @partial(jit, static_argnums=(0,))
    def _propagateInTime(
        self,
        Us,
        psis_t,
        steps_arr,
    ):
        """Propagate Full time.

        JIT compiled and uses `jax.lax.scan` to avoid compilation overhead.

        Parameters
        ----------
        psis_t : np.ndarray
            Forward propagated state
        lamdas_t : np.ndarray
            Backward propagated state
        """

        def ForwardPropagation(psis_t, index):
            psis_t = Us[index] @ psis_t
            return psis_t, psis_t

        psis_t, psis_list = scan(ForwardPropagation, psis_t, steps_arr)
        return psis_list

    def propagate(self, time: np.ndarray) -> np.ndarray:
        """Loop over all desired times in time at set resolution."""
        if self._initialState is None:
            raise ConfigurationException("Initial state is not set")

        init_state = jnp.array(self._initialState, dtype=jnp.complex128)
        dt = time[1] - time[0]

        eom_func = self._model.getMatrixEOM
        eom = eom_func(time + dt / 2) * dt

        Us = vmap(self._exponentiate, in_axes=(0,))(eom)

        psis = self._propagateInTime(
            Us, init_state, jnp.arange(0, len(time), 1)
        )
        return psis

    def gradient(self, time: np.ndarray) -> np.ndarray:
        """Compute gradients using GRAPE.

        Compute the forward propagation of the initial state and
        the backward propagation of the target state.

        Psis represent the forward propagation and lamdas represent
        the backward propagation states.

        This propagation method assumes a PWC pulse as input.
        """
        if self._initialState is None:
            raise ConfigurationException("Initial state is not set")

        if self._targetState is None:
            raise ConfigurationException("Target state is not set")

        init_state = jnp.array(self._initialState, dtype=jnp.complex128)
        target_state = jnp.array(self._targetState, dtype=jnp.complex128)
        target_state = target_state.conj().T

        eom_func = self._model.getMatrixEOM
        grad_func = self._model.gradient

        dt = time[1] - time[0]

        hams = eom_func(time + dt / 2) * dt
        dH_dps = jnp.array(grad_func(time + dt / 2)) * dt

        Ugrads = []
        n_params = dH_dps.shape[1]

        dim = init_state.shape[0]

        if self._schirmerDerivative:
            exponentiating_function = self._exponentiateSchirmer
        else:
            exponentiating_function = self._exponentiateFrechet

        for i in range(n_params):
            Us, dUs = vmap(exponentiating_function, in_axes=(None, 0, 0))(
                dim, hams, dH_dps[:, i, ...]
            )
            Ugrads.append(dUs)

        Ugrads = jnp.stack(Ugrads, axis=1)

        psis, lamdas = self._ForwardAndBackwardPropagation(
            Us, init_state, target_state, jnp.arange(0, len(time), 1)
        )

        lamdas = jnp.flip(lamdas, axis=0)

        if self._saveBwdPropagatedStates:
            self._bwdPropagatedStates = jnp.transpose(
                lamdas.conj(), axes=(0, 2, 1)
            )

        # TODO - Shift indices accordingly before doing the overlap
        grads = []
        for i in range(n_params):
            grad = vmap(
                self.__sandwichOpValues, in_axes=(0, 0, 0)
            )(
                lamdas[1:],
                Ugrads[1:, i, ...],  # type: ignore
                psis[:-1],
            )  # TODO - CHECK
            grad = jnp.squeeze(grad)
            grads.append(jnp.insert(grad, 0, grad[0]))
        return psis, jnp.array(grads)
