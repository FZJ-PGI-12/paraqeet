"""Compute gradients of state/propagator by using the GRAPE QOC method."""

import copy
import math
from collections.abc import Callable
from functools import partial
from typing import override

import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import expm_frechet

from paraqeet.exceptions import ConfigurationException
from paraqeet.propagation.propagation import DifferentiablePropagation, Propagation
from paraqeet.propagation.utils import construct_times, squeeze_trivial_axes
from paraqeet.quantity import Array, Float


class GRAPE(DifferentiablePropagation):
    r"""Compute gradient of any propagation using GRAPE :cite:p:`khaneja2005optimal`.

    It computes the gradient of the state overlap for a PWC pulses ansatz by using GRAPE.
    The initial state is propagated forwards and the target state backwards,
    and the gradient of every pulse pixel follows from sandwiching the gradient
    of the equation of motion between them,

        .. math::
            \langle \lambda(t) \lvert \frac{\partial U(t)}{\partial \alpha} \rvert \psi(t) \rangle.

    The derivative of the propagator :math:`U(t)` can be approximated (upto first order) with
    :math:`-i \Delta t \frac{\partial H}{\partial \alpha} U(t)` for closed system and
    :math:`-i \Delta t [\frac{\partial H}{\partial \alpha}, \rho] U(t)` for open system.

    Higher orders :cite:p:`defouquieres2011second` follow from differentiating
    :math:`U = \exp(\mathcal{L})` term by term, with :math:`\mathcal{L}` the generator of one
    pulse pixel and :math:`\partial\mathcal{L}` its derivative,

        .. math::
            \frac{\partial U}{\partial \alpha} = \sum_{m \geq 1} \frac{1}{m!}
            \sum_{j=0}^{m-1} \mathcal{L}^j \, \partial\mathcal{L} \, \mathcal{L}^{m-1-j}.

    Summed to all orders this is the Frechet derivative of the matrix exponential
    :cite:p:`al2009computing`, :math:`\int_0^1 e^{s \mathcal{L}} \, \partial\mathcal{L} \,
    e^{(1 - s) \mathcal{L}} \mathrm{d}s`. The Frechet derivative method can be used by setting the
    attribute `frechet_derivative` to True.

    The series is truncated after ``order`` terms. Rather than building the powers
    :math:`\mathcal{L}^j` as matrices, they are applied to the states, using
    :math:`\text{Tr}(\sigma^\dagger \mathcal{L}(X)) = \text{Tr}(\mathcal{L}^\dagger(\sigma)^\dagger X)`,

        .. math::
            \frac{\partial F}{\partial \alpha_k} = \sum_{m=1}^{\text{order}} \frac{1}{m!}
            \sum_{j=0}^{m-1} \langle (\mathcal{L}^\dagger)^j \sigma_{k+1} \lvert
            \partial\mathcal{L} \rvert \mathcal{L}^{m-1-j} \psi_k \rangle.

    ``order=1`` is textbook first-order GRAPE.

    Note:
        Each order costs one more application of the generator per pulse pixel on either side,
        and ``order*(order + 1)/2`` sandwiches per parameter. The gain stops once the truncation
        drops below the error of the propagation itself, which for an ODE solver such as
        :class:`~paraqeet.propagation.vern7.Vern7` happens beyond the second order.

    Both the forward and backward propagations are delegated to the ``_propagate`` method of *any*
    :class:`~paraqeet.propagation.propagation.Propagation`.
    The backward propagation is constructed from a copy of the original propagation class as to
    ensure the ``_propagate`` method is recompiled after changing the EOM to the adjoint EOM.

    Note:
        Current implementation of GRAPE focuses on optimizing the code speed, and can use significant
        amount of RAM. A memory efficient version would be added in a future release.

    Attributes:
        _target_state: Target state for the backward propagation.
        _operator_sandwich_function: Function that evaluates the sandwich above.
        _reverse_step_function: Step function of the backward propagation, only needed for ODE
            solvers for with collapse operators.
        _backward_prop: Propagation object used for the backward propagation (constructed from a copy of ``_prop``).
        _order: Order up to which the derivative of the propagator is expanded.
        _frechet_derivative: Whether the derivative of the propagator is the Frechet
            derivative of the matrix exponential instead of the truncated series.
    """

    _target_state: Array
    _operator_sandwich_function: Callable
    _reverse_step_function: Callable | None
    _backward_prop: Propagation
    _order: int
    _frechet_derivative: bool

    def __init__(
        self,
        propagation: Propagation,
        eom_gradient_func: Callable[[Array], Array],
        target_state: Array,
        operator_sandwich_function: Callable,
        reverse_step_function: Callable | None = None,
        order: int = 2,
        frechet_derivative: bool = False,
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
                Use 1 for textbook GRAPE. Ignored if ``frechet_derivative`` is set.
            frechet_derivative: If True use Frechet derivative instead of the expansion. Defaults to False.
        """
        super().__init__(propagation, eom_gradient_func)
        self.target_state = target_state
        self._operator_sandwich_function = operator_sandwich_function
        self._reverse_step_function = reverse_step_function
        self._backward_prop = self._create_backward_propagation()
        self.order = order
        self._frechet_derivative = frechet_derivative

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

    @property
    def frechet_derivative(self) -> bool:
        """Return whether the propagator of a pulse pixel is differentiated exactly."""
        return self._frechet_derivative

    @frechet_derivative.setter
    def frechet_derivative(self, frechet_derivative: bool) -> None:
        r"""Set whether the propagator of a pulse pixel is differentiated exactly.

        Note:
            The Frechet derivative cannot be used for ODE solver for open systems
            as they return the Hamiltonian instead of the Lindbladian.
        """
        self._frechet_derivative = frechet_derivative

    @staticmethod
    def _dagger(operators: Array) -> Array:
        """Return the adjoint of a stack of operators, with the stacking axes leading."""
        return jnp.conj(jnp.swapaxes(operators, -1, -2))

    @staticmethod
    @partial(jit, static_argnums=(0,))
    def _frechet_sandwich_func(
        operator_sandwich_function: Callable, eom: Array, eom_grads: Array, forward: Array, adjoint: Array
    ) -> Array:
        """Sandwich the Frechet derivative of the propagator of every pulse pixel, for every parameter.

        Args:
            operator_sandwich_function: Function that evaluates the matrix element of an operator.
            eom: EOM of every pulse pixel, scaled with the width of a pixel.
            eom_grads: Derivative of that EOM, with the parameter along the second axis.
            forward: Forward propagated state of every pulse pixel.
            adjoint: Adjoint of the backward propagated state of every pulse pixel.

        Returns:
            The gradient with the parameter along the first and the pulse pixel along the second axis.
        """

        def propagator_gradient(generator: Array, direction: Array) -> Array:
            derivative: Array = expm_frechet(generator, direction)[1]
            return derivative

        # Vectorized over the pulse pixel, which both arrays carry along their first axis, and
        # over the parameter, which only the direction of the derivative carries.
        propagator_grads = vmap(vmap(propagator_gradient, in_axes=(None, 0)), in_axes=(0, 0))(eom, eom_grads)
        sandwiches = vmap(operator_sandwich_function, in_axes=(1, None, None))(propagator_grads, forward, adjoint)
        return squeeze_trivial_axes(sandwiches)

    @staticmethod
    def _adjoint_eom(eom: Array, axis: int = 0) -> Array:
        """Return the adjoint EOM, in reverse order along the given axis.

        Args:
            eom: Equation of motion, sampled along ``axis``.
            axis: Axis along which the samples of one segment are stacked.
        """
        return GRAPE._dagger(jnp.flip(eom, axis=axis))

    def _apply_generator(self, propagation: Propagation, states: Array, eom: Array, dt: Float) -> Array:
        """Apply the generator of every pulse pixel to the state of that pixel.

        For propagation methods such as ODE solvers that use a ``step_function``, the generators are
        applied onto the state using the ``step_function``. Else the EOM is used to apply the generators
        onto the state.

        Note:
            This assumes that the *collapse_operators* are passed to the propagation by using the
            ``_propagate_args`` method.

        Args:
            propagation: Propagation whose ``step function`` is used. Pass the forward propagation to
                apply the generator and ``_backward_prop`` to apply its adjoint.
            states: States of every pulse pixel, with the pulse pixel along the first axis.
            eom: EOM of every pulse pixel, scaled with the width of a pixel.
            dt: Width of one pulse pixel, which scales the collapse operators.

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

    def _propagate_forward(self, times: Array) -> tuple[Array, Array, Float, Array, Array]:
        """Propagate the initial state forward and collect the EOM of every segment.

        The EOM of all segments is evaluated in a single call and all segments are propagated in
        a single compiled loop, see ``Propagation._sample_eom_batched``.

        Args:
            times: Array of times.

        Returns:
            The forward propagated states, one per entry of ``times``, the EOM with the segment
            along the first axis, the step size, the iteration indices of one segment and the
            sample times. The latter four feed the backward propagation and the gradient.

        Raises:
            ConfigurationException: If the given times are not uniformly spaced. GRAPE stacks the
                EOM of every segment along one axis, which needs them to be of equal length.
        """
        sampled_eom = self._prop._sample_eom_batched(times)
        if sampled_eom is not None:
            eom, dt, steps, sample_times = sampled_eom
            psis = self._prop._propagate_batched(eom, self._prop.initial_state, steps, *self._prop._propagate_args(dt))
            return psis, eom, dt, steps, sample_times

        # Fallback to the ``_propagate`` method, one segment at a time.
        first_step_times, dt = construct_times(times, 1, self._prop.resolution)
        steps = jnp.arange(0, len(first_step_times), 1)
        propagate_args = self._prop._propagate_args(dt)

        states: list[Array] = [self._prop.initial_state]
        eom_per_segment: list[Array] = []
        times_per_segment: list[Array] = []

        for ti in range(1, len(times)):
            step_times, dt = construct_times(times, ti, self._prop.resolution)
            times_of_segment = self._prop._construct_time_grid(step_times, dt)
            eom_of_segment = self._prop._eom_func(times_of_segment) * dt

            states.append(self._prop._propagate(eom_of_segment, states[-1], steps, *propagate_args))
            eom_per_segment.append(eom_of_segment)
            times_per_segment.append(times_of_segment)

        return (
            jnp.stack(states),
            jnp.stack(eom_per_segment),
            dt,
            steps,
            jnp.stack(times_per_segment),
        )

    def _propagate_backward(self, eom: Array, dt: Float, steps: Array) -> Array:
        """Propagate the target state backwards through the segments of the forward pass.

        The segments are reversed against each other and the samples within a segment are
        reversed among themselves, so that the backward propagation runs in reverse time.

        Args:
            eom: Equation of motion of every segment, as returned by ``_propagate_forward``.
            dt: Length of one propagation step.
            steps: Iteration indices of one segment.

        Returns:
            The backward propagated states, one per time point and in forward time order.
        """
        adjoint_eom = jnp.flip(GRAPE._adjoint_eom(eom, axis=1), axis=0)
        propagate_args = self._backward_prop._propagate_args(dt)

        if self._prop.batched_propagation:
            lamdas = self._backward_prop._propagate_batched(adjoint_eom, self._target_state, steps, *propagate_args)
            return jnp.flip(lamdas, axis=0)

        # Fallback to the ``_propagate`` method, one segment at a time.
        reverse_lamdas = [self._target_state]
        for eom_of_segment in adjoint_eom:
            reverse_lamdas.append(
                self._backward_prop._propagate(eom_of_segment, reverse_lamdas[-1], steps, *propagate_args)
            )

        return jnp.flip(jnp.stack(reverse_lamdas), axis=0)

    def _frechet_gradients(self, eom: Array, eom_grads: Array, forward: Array, backward: Array) -> Array:
        r"""Return the gradient of every parameter from the Frechet derivative of the propagator.

        Args:
            eom: EOM of every pulse pixel, scaled with the width of a pixel.
            eom_grads: Derivative of that EOM, with the parameter along the second axis.
            forward: Forward propagated state of every pulse pixel.
            backward: Backward propagated state of every pulse pixel.
        """
        # The sandwich function expects the adjoint of the backward propagated states, taken
        # over the last two axes because time is the leading one.
        adjoint = GRAPE._dagger(backward)

        gradients: Array = GRAPE._frechet_sandwich_func(
            self._operator_sandwich_function, eom, eom_grads, forward, adjoint
        )
        return gradients

    def _expansion_gradients(self, eom: Array, eom_grads: Array, forward: Array, backward: Array, dt: Float) -> Array:
        """Return the gradient of every parameter from the series expansion of the derivative of the propagator.

        The expansion is truncated after ``order`` terms.

        Args:
            eom: EOM of every pulse pixel, scaled with the width of a pixel.
            eom_grads: Derivative of that EOM, with the parameter along the second axis.
            forward: Forward propagated state of every pulse pixel.
            backward: Backward propagated state of every pulse pixel.
            dt: Width of one pulse pixel.
        """
        # Powers of the generator applied to the states of every pulse pixel, the forward states
        # carrying the generator and the backward states its adjoint.
        forward_states: list[Array] = [forward]
        backward_states: list[Array] = [backward]
        for _ in range(self._order - 1):
            forward_states.append(self._apply_generator(self._prop, forward_states[-1], eom, dt))
            backward_states.append(
                self._apply_generator(self._backward_prop, backward_states[-1], GRAPE._dagger(eom), dt)
            )

        # The sandwich function expects the adjoint of the backward propagated states, taken
        # over the last two axes because time is the leading one.
        adjoint_states = [GRAPE._dagger(states) for states in backward_states]

        # Sandwich every parameter at once, the parameter being the second axis of the derivative.
        sandwich = vmap(self._operator_sandwich_function, in_axes=(1, None, None))
        terms = [
            sandwich(eom_grads, forward_states[m - 1 - j], adjoint_states[j]) / math.factorial(m)
            for m in range(1, self._order + 1)
            for j in range(m)
        ]

        return squeeze_trivial_axes(sum(terms))

    def _sample_eom_and_grad_at_midpoint(
        self, midpoints: Array, dt: Float, segment_eom: Array, sample_times: Array
    ) -> tuple[Array, Array]:
        """Return the EOM and its derivative at the midpoint of every pulse pixel.

        For cases where the EOM and its gradient are sampled not at the midpoint (such as ODE solvers),
        this method returns the EOM and gradient at the midpoints. Else, it returns the the input EOM, and
        samples the gradient at pixel midpoints.

        Args:
            midpoints: Times in the middle of every pulse pixel.
            dt: Width of one pulse pixel.
            segment_eom: EOM of every segment as sampled by the propagation.
            sample_times: Times at which ``segment_eom`` was sampled.

        """
        samples_the_midpoint = (sample_times.shape[1] == 1) and (
            bool(jnp.allclose(sample_times[:, 0], midpoints, rtol=1e-12, atol=0.0))
        )

        if samples_the_midpoint:
            # ``segment_eom`` is already sampled at the midpoint.
            return segment_eom[:, 0], jnp.array(self._eom_gradient_func(midpoints)) * dt

        eom, eom_gradient = self._eom_and_gradient_func(midpoints)
        return jnp.array(eom) * dt, jnp.array(eom_gradient) * dt

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
            dimension and the pulse pixel along the second.

        """
        if len(times) < 2:
            raise ValueError("GRAPE.get_value_and_gradient needs at least two time points.")

        if self._prop.initial_state is None:
            raise ConfigurationException("Initial state is not set")
        if self._target_state is None:
            raise ConfigurationException("Target state is not set")

        psis, segment_eom, step_dt, steps, sample_times = self._propagate_forward(times)
        lamdas = self._propagate_backward(segment_eom, step_dt, steps)

        # The EOM and its gradient of a pulse pixel are the ones in the middle of that pixel.
        dt = times[1] - times[0]
        midpoints = times[:-1] + dt / 2
        eom, eom_grads = self._sample_eom_and_grad_at_midpoint(midpoints, dt, segment_eom, sample_times)

        forward = psis[:-1]
        backward = lamdas[1:]

        if self._frechet_derivative:
            grads = self._frechet_gradients(eom, eom_grads, forward, backward)
        else:
            grads = self._expansion_gradients(eom, eom_grads, forward, backward, dt)

        return psis, grads

    @override
    def get_gradient(self, times: Array) -> Array:
        """Return the gradient of the propagated state/propagator w.r.t. the specified parameters.

        Args:
            times: Array of times.
        """
        _, gradient = self.get_value_and_gradient(times)
        return gradient
