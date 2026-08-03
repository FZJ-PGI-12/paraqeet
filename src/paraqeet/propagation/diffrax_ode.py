"""Propagation using Diffrax ODE solver."""

from collections.abc import Callable
from functools import partial
from typing import Any, override

import diffrax
import jax
import jax.numpy as jnp
from jax import jit

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

    The equation of motion is sampled on a fixed grid and interpolated with a cubic Hermite spline in between.
    The solver therefore integrates in units of propagation steps: one unit of the solver's internal time
    is one step of length ``1 / resolution``.
    Since Diffrax is written in JAX, ``_propagate`` stays compatible with
    :class:`~paraqeet.propagation.auto_diff_gradients.AutoDiffGradients`.

    The EOM and the state are complex, but is split into the real and imaginary part for compatibility with diffrax.
    This is also what makes implicit solvers such as ``diffrax.Kvaerno5()`` usable.

    Note:
        The default configuration, ``Tsit5`` with a constant step size of one propagation step,
        reproduces the behavior of the other fixed step methods. For a high order ODE solver
        use ``solver=diffrax.Dopri8()``, and for adaptive stepping pass a
        ``stepsize_controller=diffrax.PIDController(rtol=..., atol=...)``.

    Note:
        Unlike ``Vern7``, which evaluates the EOM exactly at fixed time steps, this
        solver only sees the interpolated EOM. With the default of one sample per step the
        interpolation, the interpolation limits the accuracy to about third order in the step size,
        so ``Vern7`` is more accurate at the same resolution. Increase ``samples_per_step`` or the
        ``resolution`` if the propagation has to be more accurate than that.
    """

    _step_function: Callable
    _jump_operators: Array
    _solver: diffrax.AbstractSolver
    _stepsize_controller: diffrax.AbstractStepSizeController
    _adjoint: diffrax.AbstractAdjoint
    _dt0: float
    _max_steps: int | None
    _samples_per_step: int

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
            samples_per_step: Number of times the EOM is sampled per propagation step. Increasing
                it improves the interpolation of the EOM between the grid points at the cost of
                more EOM evaluations.

        Raises:
            ConfigurationException: If fewer than one sample per propagation step is requested.
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

        The EOM and the state are split into their real and imaginary parts before the solve and
        rejoined afterwards, so that Diffrax integrates a purely real system.

        Args:
            eom: EOM sampled at the times of ``_construct_time_grid``.
            state_t: State/propagator at the start of the interval.
            steps_arr: Iteration indices, one per propagation step.
            col: Jump operators scaled with the square root of the step size.
        """
        num_steps = steps_arr.shape[0]
        sample_times = jnp.arange(num_steps * self._samples_per_step + 1) / self._samples_per_step
        eom_split = jnp.stack([eom.real, eom.imag], axis=1)
        eom_interp = diffrax.CubicInterpolation(
            sample_times, diffrax.backward_hermite_coefficients(sample_times, eom_split)
        )

        def vector_field(time: Any, state: Array, _args: Any) -> Array:
            eom_t = eom_interp.evaluate(time)
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
