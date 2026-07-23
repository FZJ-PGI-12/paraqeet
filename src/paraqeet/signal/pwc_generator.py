"""Piecewise-constant (PWC) pulse generator used for GRAPE-style optimization."""

from functools import partial
from typing import override

import jax.numpy as jnp
from jax import jit
from jax.scipy.special import erf

from paraqeet.quantity import Array, Quantity
from paraqeet.signal.generator import Generator
from paraqeet.signal.signal import Signal


class PWCGenerator(Generator):
    """Convert a complex envelope to PWC pulse.

    This sets the pulse parameters to the `tlist` points.
    The gradient of the pulse wrt the PWC bins is 1 at that time point and zero
    everywhere else.

    This Generator doesn't add the LO signal to the envelope pulse.
    Driving with a PWC pulse (without the LO) is usually done in the rotating
    frame of drive.

    This Generator converts the input complex pulse to the 'in-phase' and
    'out-of-phase' components. This naming convention is used by following [Krantz2019].
    In the literature of signal processing these are also called 'in-phase' and 'quadrature'
    components (refer to https://en.wikipedia.org/wiki/In-phase_and_quadrature_components).

    [Krantz2019] Krantz et al., “A Quantum Engineer’s Guide to Superconducting Qubits.” Applied Physics Reviews 6(2019).

    Attributes:
        _envs: List of Envelopes
        _tlist: Left time points for discretization. These can be used for propagation and optimization.
        _time_grid: Time grid used to discretize the pulse.
            These are shifted from tlist by dt, and doesn't include zero time.
        _max_amplitude: Maximum amplitude of the drive
        _inphase: The in-phase component of the pulse
        _outofphase: The out-of-phase component of the pulse
        _optimizable_parameters: List of own parameters that would be optimized by the optimizer.
        _multiply_flat_top: Flag to multiply flat-top-Gaussain pulse to the signal to ensure it starts and ends at zero.
    """

    _envs: list[Signal]
    _tlist: Array
    _time_grid: Array
    _max_amplitude: float
    _inphase: Quantity
    _outofphase: Quantity
    _optimizable_parameters: list[Quantity] = []
    _multiply_flat_top: bool = False

    def __init__(
        self,
        envelopes: list[Signal] | None,
        tlist: Array,
        max_amplitude: float | None = None,
    ) -> None:
        """
        Args:
            envelopes: List of Envelopes
            tlist: Left time points for discretization. These can be used for propagation and optimization.
            max_amplitude: Maximum amplitude of the drive.
        """
        self._envs = envelopes or []
        self._tlist = tlist

        # Choose the center point for time grid
        dt = tlist[1] - tlist[0]
        self._time_grid = tlist[:-1] + dt / 2

        if max_amplitude is not None and max_amplitude < 0.0:
            raise ValueError("The maximum drive amplitude must be positive.")
        elif max_amplitude is None:
            env = self._compute_shape()
            self._max_amplitude = 2 * float(jnp.max(jnp.abs(env)))
        else:
            self._max_amplitude = max_amplitude

        self._setup_inphase_and_outofphase()
        self._t_final = self._time_grid[-1]

    @partial(jit, static_argnums=(0,))
    def _compute_envelope(self, t: Array) -> Array:
        t_final = self._t_final
        ramp_time = t_final / 25
        ramp_up = 1 + erf((t - 2 * t_final / 20) / ramp_time)
        ramp_down = 1 + erf((-t + 18 * t_final / 20) / ramp_time)
        return ramp_up * ramp_down / 4

    @property
    def tlist(self) -> Array:
        """Get time grid discretization for generating PWC pulse.

        Returns:
            Array of time points at which envelope is discretized.
        """
        return self._tlist

    @tlist.setter
    def tlist(self, tlist: Array) -> None:
        """Set time grid discretization for generating PWC pulse.

        Args:
            tlist: Array of time points at which envelope is discretized.
        """
        self._tlist = tlist

        # update time grid
        dt = tlist[1] - tlist[0]
        self._time_grid = tlist[:-1] + dt / 2

        self._setup_inphase_and_outofphase()

    @property
    def max_amplitude(self) -> float:
        """Get the maximum drive amplitude.

        Returns:
            The value of the maximum drive amplitude.
        """
        return self._max_amplitude

    @max_amplitude.setter
    def max_amplitude(self, max_amplitude: float) -> None:
        """Set the maximum drive amplitude of the drive.

        Args:
            max_amplitude: The value of the maximum drive amplitude.
        """
        self._max_amplitude = max_amplitude
        self._setup_inphase_and_outofphase()

    @property
    def envs(self) -> list[Signal]:
        """Gets the list of envelopes.

        Returns:
            The list of signals associated with the generator.
        """
        return self._envs

    @property
    def multiply_flat_top(self) -> bool:
        """Flag to multiply the pulse with a FlatTop.

        This can be used to make the start and end values zeros and force the
        PWC pulse to change smoothly.

        Returns:
            Flag value for multiply_flat_top.
        """
        return self._multiply_flat_top

    @multiply_flat_top.setter
    def multiply_flat_top(self, multiply_flat_top: bool) -> None:
        """Set flag to multiply the pulse with a FlatTop.

        This can be used to make the start and end values zeros and force the
        PWC pulse to change smoothly.

        Args:
            multiply_flat_top: Flag value for multiply_flat_top.
        """
        self._multiply_flat_top = multiply_flat_top
        self._setup_inphase_and_outofphase()

    def get_number_of_pwc_pixels(self) -> int:
        """Return number of PWC pixels generated by the PWC generator."""
        return len(self.tlist)

    def _compute_shape(self) -> Array:
        env = jnp.zeros_like(self._time_grid, dtype=jnp.complex128)
        for dev in self._envs:
            env += dev.get_value(self._time_grid)
        return env

    def _setup_inphase_and_outofphase(self) -> None:
        """Generate inphase and outphase Quantities using tlist."""
        env = self._compute_shape()

        # max_abs = jnp.max(jnp.abs(env))
        max_component = self._max_amplitude / jnp.sqrt(2)
        bound = max_component * jnp.ones_like(self._time_grid)

        self._inphase = Quantity(
            jnp.real(env),
            min_value=-bound,
            max_value=bound,
            name="Inphase",
        )
        self._outofphase = Quantity(
            jnp.imag(env),
            min_value=-bound,
            max_value=bound,
            name="out-of-phase",
        )

    def _get_partial_derivatives(self, times: Array) -> Array:
        env_grads = []
        for dev in self._envs:
            _, grad = dev.get_value_and_gradient(times)
            dev_grad = jnp.concat([jnp.real(grad), jnp.imag(grad)])
            env_grads.append(dev_grad)
        return jnp.hstack(env_grads)

    def _update_inphase_and_outofphase(self) -> None:
        env = self._compute_shape()
        self._inphase.set_value(jnp.real(env))
        self._outofphase.set_value(jnp.imag(env))

    @override
    def get_parameters(self) -> list[Quantity]:
        """Return a list of parameters.

        Return the inphase and out-of-phase as parameters.

        Returns:
            All Parameters describing the signal.
        """
        return [self._inphase, self._outofphase]

    @override
    def set_optimizable_parameters(self, params: list[Quantity]) -> None:
        """Set specified parameters to be optimized.

        Optimizable parameters can be inphase and out-of-phase.

        Args:
            params: Input list of parameters to be set.
        """
        super().set_optimizable_parameters(params)

    @partial(jit, static_argnums=(0,))
    def _pwc_signal(
        self,
        inphase: Array,
        outofphase: Array,
        dt: Array,
        times: Array,
    ) -> Array:
        """Generate a PWC signal.

        The PWC signal is generated by finding the closest time point
        and returning the corresponding amplitude value.

        Args:
            inphase: 1-D vector of step values of real part of the PWC signal.
            outofphase: 1-D vector of step values of complex part of the PWC signal.
            dt: Time step.
            times: Array of times.

        Returns:
            The PWC signal value at times.

        """
        index = jnp.array(times / dt, int)
        return inphase[index] + 1.0j * outofphase[index]

    @override
    def get_value(self, times: Array) -> Array:
        """Generate the PWC signal.

        Args:
            times: Array of times.

        Returns:
            The signal vector.

        """
        time_grid = self._time_grid
        dt = time_grid[1] - time_grid[0]
        inphase = self._inphase.get_value()
        outofphase = self._outofphase.get_value()
        if self._multiply_flat_top:
            env = self._compute_envelope(time_grid)
            inphase *= env
            outofphase *= env
        idx = jnp.array(times / dt, int)
        return inphase[idx] + 1j * outofphase[idx]

    @override
    def get_gradient(self, times: Array) -> Array:
        """Return signal gradient wrt inphase and out-of-phase.

        This returns a list of ones as the gradient of the envelope wrt a step
        is 1 for that time bin and 0 everywhere else.

        Args:
            times: Array of times.

        Returns:
            PWC signal gradients.
        """
        grads = []
        time_grid = self._time_grid

        if self._multiply_flat_top:
            smoothing = self._compute_envelope(time_grid)
            index = jnp.argmin(jnp.abs(jnp.expand_dims(time_grid, axis=1) - times), axis=0)
            env = smoothing[index]
        else:
            env = jnp.ones_like(times)

        if self._is_optimized(self._inphase):
            grads.append(env)
        if self._is_optimized(self._outofphase):
            grads.append(1j * env)

        if len(grads) > 0:
            grads_stack = jnp.stack(grads, axis=1)
        else:
            grads_stack = jnp.empty((times.shape[0], 0))
        return grads_stack
