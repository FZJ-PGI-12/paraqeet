"""Compute gradients of state/propagator by using the GRAPE QOC method."""

import copy
import math
from collections.abc import Callable
from typing import override

import jax.numpy as jnp
from jax import vmap

from paraqeet.exceptions import ConfigurationException
from paraqeet.propagation.propagation import DifferentiablePropagation, Propagation
from paraqeet.propagation.utils import construct_times
from paraqeet.quantity import Array, Float


class GRAPE(DifferentiablePropagation):
    r"""Compute gradient of any propagation using GRAPE :cite:p:`khaneja2005optimal`.

    It computes the gradient of the state overlap for a PWC pulses ansatz by using GRAPE.
    The initial state is propagated forwards and the target state backwards,
    and the gradient of every pulse piece follows from sandwiching the gradient
    of the equation of motion between them,

        .. math::
            \langle \lambda(t) \lvert \frac{\partial U(t)}{\partial \alpha} \rvert \psi(t) \rangle.

    The derivative of the propagator :math:`U(t)` can be approximated (upto first order) with
    :math:`-i \Delta t \frac{\partial H}{\partial \alpha} U(t)` for closed system and
    :math:`-i \Delta t [\frac{\partial H}{\partial \alpha}, \rho] U(t)` for open system.

    Higher orders :cite:p:`defouquieres2011second` follow from differentiating
    :math:`U = \exp(\mathcal{L})` term by term, with :math:`\mathcal{L}` the generator of one
    pulse piece and :math:`\partial\mathcal{L}` its derivative,

        .. math::
            \frac{\partial U}{\partial \alpha} = \sum_{m \geq 1} \frac{1}{m!}
            \sum_{j=0}^{m-1} \mathcal{L}^j \, \partial\mathcal{L} \, \mathcal{L}^{m-1-j}.

    Summed to all orders this is the Frechet derivative of the matrix exponential
    :cite:p:`al2009computing`, :math:`\int_0^1 e^{s \mathcal{L}} \, \partial\mathcal{L} \,
    e^{(1 - s) \mathcal{L}} \mathrm{d}s`.

    The series is truncated after ``order`` terms. Rather than building the powers
    :math:`\mathcal{L}^j` as matrices, they are applied to the states, using
    :math:`\text{Tr}(\sigma^\dagger \mathcal{L}(X)) = \text{Tr}(\mathcal{L}^\dagger(\sigma)^\dagger X)`,

        .. math::
            \frac{\partial F}{\partial \alpha_k} = \sum_{m=1}^{\text{order}} \frac{1}{m!}
            \sum_{j=0}^{m-1} \langle (\mathcal{L}^\dagger)^j \sigma_{k+1} \lvert
            \partial\mathcal{L} \rvert \mathcal{L}^{m-1-j} \psi_k \rangle.

    ``order=1`` is textbook first-order GRAPE.

    Note:
        Each order costs one more application of the generator per pulse piece on either side,
        and ``order*(order + 1)/2`` sandwiches per parameter. The gain stops once the truncation
        drops below the error of the propagation itself, which for an ODE solver such as
        :class:`~paraqeet.propagation.vern7.Vern7` happens beyond the second order.

    Both the forward and backward propagations are delegated to the ``_propagate`` method of *any*
    :class:`~paraqeet.propagation.propagation.Propagation`.
    The backward propagation is constructed from a copy of the original propagation class as to
    ensure the ``_propagate`` method is recompiled after changing the EOM to the adjoint EOM.

    Attributes:
        _target_state: Target state for the backward propagation.
        _operator_sandwich_function: Function that evaluates the sandwich above.
        _reverse_step_function: Step function of the backward propagation, only needed for ODE
            solvers for with collapse operators.
        _backward_prop: Propagation object used for the backward propagation (constructed from a copy of ``_prop``).
        _order: Order up to which the derivative of the propagator is expanded.
    """

    _target_state: Array
    _operator_sandwich_function: Callable
    _reverse_step_function: Callable | None
    _backward_prop: Propagation
    _order: int

    def __init__(
        self,
        propagation: Propagation,
        eom_gradient_func: Callable[[Array], Array],
        target_state: Array,
        operator_sandwich_function: Callable,
        reverse_step_function: Callable | None = None,
        order: int = 2,
    ):
        """
        Args:
            propagation: Any propagation object.
            eom_gradient_func: Function that returns the gradient of EOM.
            target_state: Target state to be propagated backwards.
            operator_sandwich_function: Operator sandwich function to compute GRAPE gradients.
                Use ``grape_operator_sandwich_function_closed`` for a closed system and
                ``grape_operator_sandwich_function_open`` for an open one.
            reverse_step_function: Step function of the backward propagation. Only ODE solvers
                that build a dissipator from collapse operators need one, use
                ``reverse_lindblad_step`` for the Lindblad master equation in density matrix form.
            order: Order up to which the derivative of the propagator is expanded. Defaults to 2.
                Use 1 for textbook GRAPE.
        """
        super().__init__(propagation, eom_gradient_func)
        self.target_state = target_state
        self._operator_sandwich_function = operator_sandwich_function
        self._reverse_step_function = reverse_step_function
        self._backward_prop = self._create_backward_propagation()
        self.order = order

    def _create_backward_propagation(self) -> Propagation:
        """Return the propagation object that propagates the target state backwards.

        Without a ``reverse_step_function`` this is the forward propagation itself, applied to the
        adjoint equation of motion. With one, it is a copy of the forward propagation that carries
        the reverse step function.

        Note:
            The step function of the forward propagation must not be swapped in place for the
            backward pass. Propagation methods in this package compile the ``_propagate`` method
            using JIT with ``self`` as a static argument. Hence, swapping the step_function does not
            work unless a compilation is triggered. Thus, we create a copy of the propagation object for
            the backward propagation such as to recompile the function.

        Raises:
            ConfigurationException: If the propagation carries collapse operators but no
                ``reverse_step_function`` was given, or if a ``reverse_step_function`` was given
                for a propagation that does not use a step function.
        """
        collapse_operators = getattr(self._prop, "jump_operators", None)
        is_dissipative = collapse_operators is not None and len(collapse_operators) > 0

        if self._reverse_step_function is None:
            if is_dissipative:
                raise ConfigurationException(
                    "The propagation carries collapse operators, so the backward propagation of GRAPE needs "
                    "the adjoint of the dissipator. Specify a ``reverse_step_function``, "
                    "for example ``reverse_lindblad_step``."
                )
            return self._prop

        if not hasattr(self._prop, "step_function"):
            raise ConfigurationException(
                "A ``reverse_step_function`` was specified, but the propagation does not use a step function."
            )

        backward_prop = copy.copy(self._prop)
        backward_prop.step_function = self._reverse_step_function  # type: ignore[attr-defined]
        return backward_prop

    @property
    def target_state(self) -> Array:
        """Return the target state of the backward propagation."""
        return self._target_state

    @target_state.setter
    def target_state(self, target_state: Array) -> None:
        """Set the target state of the backward propagation."""
        # TODO: Provide explicit wrappers for multiple initial states or density vectors
        self._target_state = jnp.array(target_state, dtype=jnp.complex128)

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
        """Set the operator sandwich function for computing the gradients."""
        self._operator_sandwich_function = operator_sandwich_func

    @property
    def order(self) -> int:
        """Return the order up to which the derivative of the propagator is expanded."""
        return self._order

    @order.setter
    def order(self, order: int) -> None:
        """Set the order up to which the derivative of the propagator is expanded.

        Raises:
            ConfigurationException: If the order is smaller than one.
        """
        if order < 1:
            raise ConfigurationException("The order of the GRAPE gradient has to be at least one.")
        self._order = order

    @staticmethod
    def _dagger(operators: Array) -> Array:
        """Return the adjoint of a stack of operators, with the stacking axes leading."""
        return jnp.conj(jnp.swapaxes(operators, -1, -2))

    @staticmethod
    def _adjoint_eom(eom: Array) -> Array:
        """Return the adjoint EOM of one interval, in reverse order."""
        return GRAPE._dagger(jnp.flip(eom, axis=0))

    def _apply_generator(self, propagation: Propagation, states: Array, eom: Array, dt: Float) -> Array:
        """Apply the generator of every pulse piece to the state of that piece.

        For propagation methods such as ODE solvers that use a ``step_function``, the generators are
        applied onto the state using the ``step_function``. Else the EOM is used to apply the generators
        onto the state.

        Note:
            This assumes that the *collapse_operators* are passed to the propagation by using the
            ``_propagate_args`` method.

        Args:
            propagation: Propagation whose ``step function`` is used. Pass the forward propagation to
                apply the generator and ``_backward_prop`` to apply its adjoint.
            states: States of every pulse piece, with the pulse piece along the first axis.
            eom: EOM of every pulse piece, scaled with the width of a piece.
            dt: Width of one pulse piece, which scales the collapse operators.

        Returns:
            The states with the generator applied, in the shape of ``states``.
        """
        step_function = getattr(propagation, "step_function", None)
        if step_function is None:
            product: Array = jnp.matmul(eom, states)
            return product

        step_args = propagation._propagate_args(dt)
        applied: Array = vmap(step_function, in_axes=(0, 0) + (None,) * len(step_args))(states, eom, *step_args)
        return applied

    def _propagate_forward(self, times: Array) -> tuple[list[Array], list[tuple[Array, Float, int]]]:
        """Propagate the initial state forward and collect the EOM of every interval.

        Args:
            times: Array of times.

        Returns:
            The forward propagated states, one per entry of ``times``, and per interval the
            equation of motion, the step size and the number of propagation steps (needed for backward propagation).
        """
        psis = [self._prop.initial_state]
        intervals: list[tuple[Array, Float, int]] = []

        for ti in range(1, len(times)):
            step_times, dt = construct_times(times, ti, self._prop.resolution)
            eom = self._prop._eom_func(self._prop._construct_time_grid(step_times, dt)) * dt
            psis.append(
                self._prop._propagate(
                    eom,
                    psis[-1],
                    jnp.arange(0, len(step_times), 1),
                    *self._prop._propagate_args(dt),
                )
            )
            intervals.append((eom, dt, len(step_times)))

        return psis, intervals

    def _propagate_backward(self, intervals: list[tuple[Array, Float, int]]) -> list[Array]:
        """Propagate the target state backwards through the intervals of the forward pass.

        Args:
            intervals: Per interval the equation of motion, the step size and the number of
                propagation steps, as returned by ``_propagate_forward``.

        Returns:
            The backward propagated states, one per time point and in forward time order.
        """
        lamdas = [self._target_state]

        for eom, dt, n_steps in reversed(intervals):
            lamdas.append(
                self._backward_prop._propagate(
                    GRAPE._adjoint_eom(eom),
                    lamdas[-1],
                    jnp.arange(0, n_steps, 1),
                    *self._backward_prop._propagate_args(dt),
                )
            )

        lamdas.reverse()

        return lamdas

    @override
    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
        """Compute gradients using GRAPE.

        Compute the forward propagation of the initial state and the backward propagation of
        the target state. ``psis`` represent the forward propagation and ``lamdas`` represent
        the backward propagation states.

        This propagation method assumes a PWC pulse on a uniform time grid as input.

        Args:
            times: Array of times.

        Returns:
            A tuple ``(value, gradient)``. ``value`` holds the forward propagated states with
            time along the first dimension. ``gradient`` has the parameter along the first
            dimension and the pulse piece along the second.

        """
        if len(times) < 2:
            raise ValueError("GRAPE.get_value_and_gradient needs at least two time points.")

        if self._prop.initial_state is None:
            raise ConfigurationException("Initial state is not set")
        if self._target_state is None:
            raise ConfigurationException("Target state is not set")

        psis, intervals = self._propagate_forward(times)
        lamdas = self._propagate_backward(intervals)

        # The EOM and its gradient are evaluated once per pulse piece, in the middle of it.
        dt = times[1] - times[0]
        eom, eom_gradient = self._eom_and_gradient_func(times[:-1] + dt / 2)
        eom = jnp.array(eom) * dt
        eom_grads = jnp.array(eom_gradient) * dt

        # Powers of the generator applied to the states of every pulse piece, the forward states
        # carrying the generator and the backward states its adjoint.
        forward_states: list[Array] = [jnp.array(psis[:-1])]
        backward_states: list[Array] = [jnp.array(lamdas[1:])]
        for _ in range(self._order - 1):
            forward_states.append(self._apply_generator(self._prop, forward_states[-1], eom, dt))
            backward_states.append(
                self._apply_generator(self._backward_prop, backward_states[-1], GRAPE._dagger(eom), dt)
            )

        # The sandwich function expects the adjoint of the backward propagated states, taken
        # over the last two axes because time is the leading one.
        adjoint_states = [GRAPE._dagger(states) for states in backward_states]

        grads = []
        for ii in range(eom_grads.shape[1]):
            terms = [
                self._operator_sandwich_function(eom_grads[:, ii, ...], forward_states[m - 1 - j], adjoint_states[j])
                / math.factorial(m)
                for m in range(1, self._order + 1)
                for j in range(m)
            ]
            grads.append(jnp.squeeze(sum(terms)))

        return jnp.array(psis), jnp.array(grads)

    @override
    def get_gradient(self, times: Array) -> Array:
        """Return the gradient of the propagated state/propagator w.r.t. the specified parameters.

        Args:
            times: Array of times.
        """
        _, gradient = self.get_value_and_gradient(times)
        return gradient
