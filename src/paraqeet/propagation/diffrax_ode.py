"""Propagation using Diffrax ODE solver."""

from collections.abc import Callable
from functools import partial
from typing import Any, override

import diffrax
import jax
import jax.numpy as jnp
from jax import jit
from jax.lax import dynamic_slice_in_dim

from paraqeet.exceptions import ConfigurationException
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array, Float

jax.config.update("jax_enable_x64", True)


class DiffraxODE(Propagation):
    """
    Propagate state by solving the Schrödinger equation / Lindblad master equation with Diffrax.

    This utilizes the ODE solvers of `Diffrax <https://docs.kidger.site/diffrax/>`_ :cite:p:`kidger2021on`, which
    adds adaptive step size control and a choice of explicit and implicit solvers. Like ``Vern7``
    it takes a ``step_function`` that implements the right hand side of the equation of motion,
    so the step functions of :mod:`paraqeet.propagation.utils` can be used with both classes.

    The equation of motion is sampled on an equidistant grid and interpolated in between by a Lagrange polynomial
    through the ``interpolation_order`` nearest grid points. This is required to preserve the adaptive
    time step in Diffrax based ODE solvers.
    The solver therefore integrates in units of propagation steps: one unit of the solver's internal time
    is one step of length ``1 / resolution``.
    Since Diffrax is written in JAX, ``_propagate`` stays compatible with
    :class:`~paraqeet.propagation.auto_diff_gradients.AutoDiffGradients`.

    The EOM and the state are complex, but is split into the real and imaginary part for compatibility with diffrax.
    This is also what makes implicit solvers such as ``diffrax.Kvaerno5()`` usable.

    Note:
        For a high order ODE solver
        use ``solver=diffrax.Dopri8()`` together with a larger ``interpolation_order``, and for
        adaptive stepping pass a ``stepsize_controller=diffrax.PIDController(rtol=..., atol=...)``.

    Note:
        As this solver sees the interpolated EOM, the order of convergence is the minimum of the
        order of the solver and the ``interpolation_order``. Hence, always choose the ``interpolation_order``
        to be around the order of the solver.

    Note:
        ``samples_per_step`` decrease the effective time step and matters for the case of an
        adaptive solver. Extra samples increase the number of EOM evaluations, hence should only
        be used in case the solver does not reach its convergence.
    """

    _step_function: Callable
    _jump_operators: Array
    _solver: diffrax.AbstractSolver
    _stepsize_controller: diffrax.AbstractStepSizeController
    _adjoint: diffrax.AbstractAdjoint
    _dt0: float
    _max_steps: int | None
    _samples_per_step: int
    _interpolation_order: int

    def __init__(
        self,
        eom_func: Callable[[Array], Array],
        resolution: float,
        initial_state: Array,
        step_function: Callable,
        jump_operators: list[Array] | None = None,
        solver: diffrax.AbstractSolver | None = None,
        stepsize_controller: diffrax.AbstractStepSizeController | None = None,
        adjoint: diffrax.AbstractAdjoint | None = None,
        dt0: float = 1.0,
        max_steps: int | None = None,
        samples_per_step: int = 1,
        interpolation_order: int = 8,
    ) -> None:
        """
        Args:
            eom_func: Equation of motion (EOM) as a function of time.
            resolution: Resolution at which to sample the EOM.
            initial_state: Initial state.
            step_function: Step function that implements the right hand side of the EOM.
            jump_operators: A list of jump operators (each multiplied by the sqrt of the corresponding decay rate).
                Defaults to None for closed system.
            solver: Any Diffrax solver. Defaults to ``diffrax.Tsit5()``.
            stepsize_controller: Any Diffrax step size controller. Defaults to
                ``diffrax.ConstantStepSize()``, which together with ``dt0`` gives fixed steps.
            adjoint: How Diffrax differentiates through the solve. Defaults to
                ``diffrax.RecursiveCheckpointAdjoint()``, which supports reverse-mode AD.
            dt0: Initial step size of the solver, in units of propagation steps. With a constant
                step size controller the default of one propagation step is used throughout.
            max_steps: Maximum number of solver steps per interval of the times array. Defaults to
                None, in which case it is chosen large enough for the constant step size solve.
            samples_per_step: Number of times the EOM is sampled per propagation step. Only useful
                for adaptive ``stepsize_controller``, PWC pulse.
            interpolation_order: Number of grid points the EOM is interpolated over between the
                samples, which is the order of convergence the interpolation supports. Defaults to 8,
                enough for the fifth order default solver.

        Raises:
            ConfigurationException: If fewer than one sample per propagation step, or an interpolation
                over fewer than two grid points, is requested.
        """
        super().__init__(eom_func, resolution, initial_state)
        self._step_function = step_function
        self.jump_operators = jump_operators
        self._solver = solver if solver is not None else diffrax.Tsit5()
        self._stepsize_controller = (
            stepsize_controller if stepsize_controller is not None else diffrax.ConstantStepSize()
        )
        self._adjoint = adjoint if adjoint is not None else diffrax.RecursiveCheckpointAdjoint()
        self._dt0 = dt0
        self._max_steps = max_steps
        if samples_per_step < 1:
            raise ConfigurationException("DiffraxODE needs at least one EOM sample per propagation step.")
        self._samples_per_step = samples_per_step
        if interpolation_order < 2:
            raise ConfigurationException("DiffraxODE needs to interpolate the EOM over at least two grid points.")
        self._interpolation_order = interpolation_order

    @property
    def step_function(self) -> Callable:
        """Return the step function for solving the EOM."""
        return self._step_function

    @step_function.setter
    def step_function(self, step_func: Callable) -> None:
        """Set the step function for solving the EOM."""
        self._step_function = step_func

    @property
    def jump_operators(self) -> Array:
        """Return the jump operators used for solving the EOM."""
        return self._jump_operators

    @jump_operators.setter
    def jump_operators(self, jump_ops: list[Array] | None) -> None:
        """Set the jump operators added to the EOM."""
        if jump_ops is not None:
            self._jump_operators = jnp.array(jump_ops)
        else:
            self._jump_operators = jnp.empty((0,) + self._eom_func(jnp.array([0.0])).shape)

    @override
    def _propagate_args(self, dt: Float) -> tuple[Array, ...]:
        """Return the jump operators scaled with the step size."""
        return (self._jump_operators * jnp.sqrt(dt),)

    @override
    def _construct_time_grid(self, step_times: Array, dt: Float) -> Array:
        """Return the sample times of the EOM, ``samples_per_step`` per step plus the final time.

        The samples are equidistant, so that they are the knots of the interpolation that
        ``_propagate`` evaluates the EOM at.
        """
        offsets = jnp.arange(self._samples_per_step) * (dt / self._samples_per_step)
        times_interp = (step_times[:, None] + offsets[None, :]).reshape(-1)
        return jnp.concatenate([times_interp, step_times[-1:] + dt])

    @partial(jit, static_argnums=(0,))
    @override
    def _propagate(self, eom: Array, state_t: Array, steps_arr: Array, col: Array) -> Array:
        """
        Propagate from ``time[ti]`` to ``time[ti+1]`` with a Diffrax solver.

        The solve runs in units of propagation steps, from ``0`` to the number of steps, because
        the EOM comes in already scaled with the step size.

        Between the samples the EOM is interpolated by the Lagrange polynomial through the
        ``interpolation_order`` nearest grid points.

        The EOM and the state are split into their real and imaginary parts before the solve and
        rejoined afterwards, so that Diffrax integrates a purely real system.

        Args:
            eom: EOM sampled at the times of ``_construct_time_grid``.
            state_t: State/propagator at the start of the interval.
            steps_arr: Iteration indices, one per propagation step.
            col: Jump operators scaled with the square root of the step size.
        """
        num_steps = steps_arr.shape[0]
        num_knots = num_steps * self._samples_per_step + 1
        stencil = min(self._interpolation_order, num_knots)
        eom_split = jnp.stack([eom.real, eom.imag], axis=1)

        nodes = jnp.arange(stencil, dtype=float)
        diagonal = jnp.eye(stencil, dtype=bool)
        denominator = jnp.where(diagonal, 1.0, nodes[:, None] - nodes[None, :])

        def vector_field(time: Any, state: Array, _args: Any) -> Array:
            knot = time * self._samples_per_step
            base = jnp.clip(jnp.floor(knot).astype(int) - (stencil // 2 - 1), 0, num_knots - stencil)
            offset = knot - base
            weights = jnp.prod(jnp.where(diagonal, 1.0, (offset - nodes)[None, :] / denominator), axis=1)
            eom_t = jnp.tensordot(weights, dynamic_slice_in_dim(eom_split, base, stencil, axis=0), axes=1)
            state_t: Array = self._step_function(state[0] + 1j * state[1], eom_t[0] + 1j * eom_t[1], col)
            return jnp.stack([state_t.real, state_t.imag])

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            self._solver,
            t0=0.0,
            t1=float(num_steps),
            dt0=self._dt0,
            y0=jnp.stack([state_t.real, state_t.imag]),
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=self._stepsize_controller,
            adjoint=self._adjoint,
            max_steps=self._max_steps if self._max_steps is not None else max(4096, 4 * num_steps),
        )
        state_new: Array = solution.ys[-1]
        return state_new[0] + 1j * state_new[1]
