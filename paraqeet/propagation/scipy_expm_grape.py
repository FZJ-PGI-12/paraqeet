"""Class definition of the Scipy piecewise exponential propagation model.

Uses the GRAPE optimisation method.
Assumes that the signal is piecewise constant (PWC) without an LO and the
Hamiltonian is defined in the rotating frame of drive.

"""

from functools import partial

import jax
import jax.numpy as jnp
from paraqeet.quantity import Array
from jax import jit, vmap
from jax.lax import scan
from jax.scipy.linalg import expm, expm_frechet

from paraqeet.exceptions import ConfigurationException
from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.propagation.scipy_expm import ScipyExpm

jax.config.update("jax_enable_x64", True)


class ScipyExpmGRAPE(ScipyExpm):
    """Solve EOMs by piecewise exponentation via Scipy using GRAPE.

    Compute the gradients of a closed quantum system for PWC pulses by using
    GRAPE. Here, we use forward propagation of the initial state and backward
    propagation of the target state to compute the gradients.

    The state propagations are done by the `ScipyExpm` method.

    _res: float
        Simulation resolution.
    _initial_state: Array = None
        Initial state for forward propagation.
    _target_state: Array = None
        Target state for backward propagation.
    _schirmer_derivative: bool = False
        If true, compute the gradient by Schirmer Derivative/Method of auxillary
        matrix exponential. If false, use frechet derivative.
    """

    _target_state: Array | None = None
    _schirmer_derivative: bool = False

    def __init__(self, model: EquationOfMotion, res: float):
        super().__init__(model, res)

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
        # Verify if `model.ode_propagation` is set to `False`.
        # ode_propgation returns hamiltonian and collapse operators separately.
        if self._model is None:
            raise ConfigurationException("No equation of motion is configured.")
        eom = self._model.get_matrix(jnp.array([0]))
        if len(eom) == 2:
            raise ConfigurationException("Please set `model.ode_propagation` to `False` for this propagation method.")

        # For open system convert Density Matrix to Vectorized form.
        if self.is_open:
            try:
                if len(targetState.shape) == 1:  # An (n,) array
                    targetState = jnp.reshape(targetState, (-1, 1))
                # Compare the shapes of target state with the generator of time translation
                dim_generator = eom.shape[1]
                # Comparing dim -2 as 0 can be batch dimension
                if targetState.shape[-2] == jnp.sqrt(dim_generator):
                    # check if it is a square matrix. Check the last 2 dimensions are equal.
                    if targetState.shape[-1] == targetState.shape[-2]:
                        # This is a density matrix
                        targetState = self._convert_dm_to_vec(targetState)
            except Exception as e:
                raise ConfigurationException(
                    f"Obtained a state vector of shape {targetState.shape} as target state. "
                    + "For open system propagation expected a density matrix or vectorized density matrix "
                    + "as the target state.\n"
                    + f"Raised exception: `{e}`"
                )
        self._target_state = targetState

    @property
    def use_schirmer_derivative(self) -> bool:
        """Returns whether the Schirmer method is used to compute the derivative of the unitary operator."""
        return self._schirmer_derivative

    @use_schirmer_derivative.setter
    def use_schirmer_derivative(self, schirmerDerivative: bool) -> None:
        """Schirmer Derivative method to compute derivative of Unitary operator.

        Parameters
        ----------
        schirmerDerivative : bool
            If True use Schirmer derivative, if False use Frechet Derivative.
        """
        self._schirmer_derivative = schirmerDerivative

    @staticmethod
    @jit
    def __sandwich_op_values(
        bwd_propagated_state: Array,
        Op: Array,
        fwd_propagated_state: Array,
    ) -> Array:
        r"""Compute \\langle \\lambda(t) | O | \\psi(t) \\rangle.

        Parameters
        ----------
        bwd_propagated_state : Array
            Backwards propagated states
        Op : Array
            Array of operator for each time point.
        fwd_propagated_state : Array
            Forwards propagated states

        Returns
        -------
        Array
            Matrix element of the operator for each time point.
        """
        return jnp.matmul(bwd_propagated_state, jnp.matmul(Op, fwd_propagated_state))

    @partial(jit, static_argnums=(0,))
    def _forward_and_backward_propagation(
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
        psis_t : Array
            Forward propagated state
        lamdas_t : Array
            Backward propagated state
        """

        def forward_propagation(psis_t, index):
            psis_t = Us[index] @ psis_t
            return psis_t, psis_t

        def backward_propagation(lamdas_t, index):
            lamdas_t = lamdas_t @ Us[-index - 1]
            return lamdas_t, lamdas_t

        psis_t, psis_list = scan(forward_propagation, psis_t, steps_arr)
        lamdas_t, lamdas_list = scan(backward_propagation, lamdas_t, steps_arr)

        return psis_list, lamdas_list

    @partial(jit, static_argnums=(0,))
    def _forward_and_backward_propagation_open(
        self,
        Us,
        Us_rev,
        psis_t,
        lamdas_t,
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
            psis_t = Us[index] @ psis_t
            return psis_t, psis_t

        def backward_propagation(lamdas_t, index):
            lamdas_t = Us_rev[index] @ lamdas_t
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
        ham : Array
            -iHdt
        dh_dp : Array
            -i\\frac{\\partial H}{\\partial u} dt
        """
        return expm_frechet(ham, dh_dp)

    @staticmethod
    @partial(jit, static_argnums=(0,))
    def _exponentiate_schirmer(dim, ham, dh_dp):
        r"""Exponentiate an auxilliary matrix to compute U and dU.

        Parameters
        ----------
        ham : Array
            -iHdt
        dh_dp : Array
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
            ham : Array
            -iHdt
        """
        return expm(ham)

    @partial(jit, static_argnums=(0,))
    def _propagate_in_time(
        self,
        Us,
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
            psis_t = Us[index] @ psis_t
            return psis_t, psis_t

        psis_t, psis_list = scan(forward_propagation, psis_t, steps_arr)
        return psis_list

    def propagate(self, time: Array) -> Array:
        """Loop over all desired times in time at set resolution."""
        if self._initial_state is None:
            raise ConfigurationException("Initial state is not set")

        init_state = jnp.array(self._initial_state, dtype=jnp.complex128)
        dt = time[1] - time[0]

        timeGrid = time[:-1] + dt / 2

        if self._model is None:
            raise ConfigurationException("No model is configured to provide an equation of motion.")

        eom_func = self._model.get_matrix
        eom = eom_func(timeGrid) * dt

        Us = vmap(self._exponentiate, in_axes=(0,))(eom)

        psis = self._propagate_in_time(Us, init_state, jnp.arange(0, len(timeGrid), 1))
        psis = jnp.concat([jnp.expand_dims(init_state, axis=0), psis], axis=0)

        # if open system convert back the vectorized density matrices to matrix shape
        dim = eom.shape[-2]
        if self.is_open:
            psis = jnp.array(psis)
            psis = vmap(self._convert_vec_to_dm, in_axes=(0, None))(psis, int(jnp.sqrt(dim)))
        return jnp.array(psis)

    def __gradient_closed_system(self, time: Array) -> tuple[Array, Array]:
        init_state = jnp.array(self._initial_state, dtype=jnp.complex128)
        target_state = jnp.array(self._target_state, dtype=jnp.complex128)
        target_state = target_state.conj().T

        if self._model is None:
            raise ConfigurationException("No model is configured to provide an equation of motion.")

        eom_func = self._model.get_matrix
        grad_func = self._model.gradient

        dt = time[1] - time[0]

        timeGrid = time[:-1] + dt / 2

        hams = eom_func(timeGrid) * dt
        dH_dps = jnp.array(grad_func(timeGrid)) * dt

        Ugrads_list = []
        n_params = dH_dps.shape[1]

        dim = hams.shape[-2]

        if self._schirmer_derivative:
            exponentiating_function = self._exponentiate_schirmer
        else:
            exponentiating_function = self._exponentiate_frechet

        for i in range(n_params):
            Us, dUs = vmap(exponentiating_function, in_axes=(None, 0, 0))(dim, hams, dH_dps[:, i, ...])
            Ugrads_list.append(dUs)

        Ugrads = jnp.stack(Ugrads_list, axis=1)

        psis, lamdas = self._forward_and_backward_propagation(
            Us, init_state, target_state, jnp.arange(0, len(timeGrid), 1)
        )

        psis = jnp.concat([jnp.expand_dims(init_state, axis=0), psis], axis=0)
        lamdas = jnp.concat([jnp.expand_dims(target_state, axis=0), lamdas], axis=0)

        lamdas = jnp.flip(lamdas, axis=0)

        grads = []
        for i in range(n_params):
            grad = vmap(
                self.__sandwich_op_values, in_axes=(0, 0, 0)
            )(
                lamdas[1:],
                Ugrads[:, i, ...],  # type: ignore
                psis[:-1],
            )
            grad = jnp.squeeze(grad)
            grads.append(grad)

        return psis, jnp.array(grads)

    def __gradient_open_systems(self, time: Array) -> tuple[Array, Array]:
        raise NotImplementedError(
            "Currently ScipyExpmGRAPE is not supported for open system optimisation."
            + " Use Vern7GRAPE as an alternative (with `model.ode_propagation = True`)."
        )

    def gradient(self, time: Array) -> tuple[Array, Array]:
        """Compute gradients using GRAPE.

        Compute the forward propagation of the initial state and
        the backward propagation of the target state.

        Psis represent the forward propagation and lamdas represent
        the backward propagation states.

        This propagation method assumes a PWC pulse as input.
        """
        if self._initial_state is None:
            raise ConfigurationException("Initial state is not set")

        if self._target_state is None:
            raise ConfigurationException("Target state is not set")

        if self.is_open:
            psis, grads = self.__gradient_open_systems(time)
        else:
            psis, grads = self.__gradient_closed_system(time)

        return psis, grads
