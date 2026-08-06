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

    This sets the pulse parameters to the ``tlist`` points.
    The gradient of the pulse wrt the PWC pixels is 1 at that time point and zero
    everywhere else.

    This Generator doesn't add the LO signal to the envelope pulse.
    Driving with a PWC pulse (without the LO) is usually done in the rotating
    frame of drive.

    This Generator converts the input complex pulse to the 'in-phase' and
    'out-of-phase' components. This naming convention follows :cite:p:`krantz2019quantum`.
    In the signal processing literature, these are also called 'in-phase' and 'quadrature'
    components (refer to https://en.wikipedia.org/wiki/In-phase_and_quadrature_components).

    Attributes:
        _envs: List of Envelopes
        _tlist: Left time points for discretization. These can be used for propagation and optimization.
        _time_grid: Time grid used to discretize the pulse.
            These are shifted from tlist by dt/2 and do not include zero time.
        _max_amplitude: Maximum amplitude of the drive
        _inphase: The in-phase component of the pulse
        _outofphase: The out-of-phase component of the pulse
        _optimizable_parameters: List of own parameters that would be optimized by the optimizer.
        _multiply_flat_top: Flag to multiply flat-top-Gaussian pulse to the signal to ensure it starts and ends at zero.
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

    @staticmethod
    @jit
    def _flat_top_envelope(times: Array, t_final: Array) -> Array:
        """Return the flat-top Gaussian that forces the pulse to start and end at zero.

        Args:
            times: Array of times.
            t_final: Final time of the pulse.
        """
        ramp_time = t_final / 25
        ramp_up = 1 + erf((times - 2 * t_final / 20) / ramp_time)
        ramp_down = 1 + erf((-times + 18 * t_final / 20) / ramp_time)
        return ramp_up * ramp_down / 4

    def _compute_shape(self) -> Array:
        env = jnp.zeros_like(self._time_grid, dtype=jnp.complex128)
        for dev in self._envs:
            env += dev.get_value(self._time_grid)
        return env

    def _setup_inphase_and_outofphase(self) -> None:
        """Generate in-phase and out-of-phase Quantities using tlist."""
        env = self._compute_shape()

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

    @partial(jit, static_argnums=(0,))
    def _compute_pixel_index(self, times: Array) -> Array:
        """Return the index of the PWC pixel that every time falls into.

        A pixel covers ``[tlist[k], tlist[k + 1])``, so a time on a boundary belongs to the pixel
        starting there. Times outside the pulse are clipped to the first and the last pixel.
        Comparing against the boundaries rather than dividing by the width of a pixel keeps this
        exact for a pulse that does not start at zero and for a non-uniform ``tlist``.

        Args:
            times: Array of times.
        """
        return jnp.clip(jnp.searchsorted(self._tlist, times, side="right") - 1, 0, self._tlist.shape[0] - 2)

    def get_pixel_times(self, times: Array) -> Array:
        """Return the center of the PWC pixel whose value applies at every given time.

        The pulse is constant over a pixel, so its value at a time is the value of the envelope at
        the center of that pixel, not at the time itself.

        Args:
            times: Array of times.

        Returns:
            The pixel center for every entry of ``times``.
        """
        return self._time_grid[self._compute_pixel_index(times)]

    @staticmethod
    @partial(jit, static_argnums=(0,))
    def _compute_value(
        multiply_flat_top: bool,
        inphase: Array,
        outofphase: Array,
        time_grid: Array,
        t_final: Array,
        index: Array,
    ) -> Array:
        """Return the PWC signal of the pixels the given times fall into.

        Note:
            The mutable state is passed in rather than read off ``self``: the ``tlist``,
            ``max_amplitude`` and ``multiply_flat_top`` setters rebuild it, and a cache keyed on
            the generator object would keep serving the values of the first call.

        Args:
            multiply_flat_top: Whether the pixel amplitudes are multiplied with a flat-top Gaussian.
            inphase: In-phase amplitude of every pixel.
            outofphase: Out-of-phase amplitude of every pixel.
            time_grid: Centers of the PWC pixels.
            t_final: Final time of the pulse.
            index: Index of the pixel that every time falls into.
        """
        if multiply_flat_top:
            envelope = PWCGenerator._flat_top_envelope(time_grid, t_final)
            inphase = inphase * envelope
            outofphase = outofphase * envelope

        return inphase[index] + 1j * outofphase[index]

    @staticmethod
    @partial(jit, static_argnums=(0, 1, 2))
    def _compute_gradient(
        multiply_flat_top: bool,
        inphase_is_optimized: bool,
        outofphase_is_optimized: bool,
        time_grid: Array,
        t_final: Array,
        index: Array,
    ) -> Array:
        """Return the derivative of the PWC signal with respect to the pixel amplitudes.

        See :meth:`_compute_value` for why the mutable state is passed in as arguments.

        Args:
            multiply_flat_top: Whether the pixel amplitudes are multiplied with a flat-top Gaussian.
            inphase_is_optimized: Whether the in-phase amplitudes are optimized.
            outofphase_is_optimized: Whether the out-of-phase amplitudes are optimized.
            time_grid: Centers of the PWC pixels.
            t_final: Final time of the pulse.
            index: Index of the pixel that every time falls into.
        """
        if multiply_flat_top:
            smoothing = PWCGenerator._flat_top_envelope(time_grid, t_final)
            env = smoothing[index]
        else:
            env = jnp.ones(index.shape)

        grads = []
        if inphase_is_optimized:
            grads.append(env)
        if outofphase_is_optimized:
            grads.append(1j * env)

        if len(grads) > 0:
            return jnp.stack(grads, axis=1)
        return jnp.empty((index.shape[0], 0))

    @override
    def get_value(self, times: Array) -> Array:
        """Generate the PWC signal.

        Args:
            times: Array of times.

        Returns:
            The signal vector.

        """
        value: Array = PWCGenerator._compute_value(
            self._multiply_flat_top,
            self._inphase.get_value(),
            self._outofphase.get_value(),
            self._time_grid,
            self._t_final,
            self._compute_pixel_index(times),
        )
        return value

    @override
    def get_gradient(self, times: Array) -> Array:
        """Return signal gradient wrt inphase and out-of-phase.

        This returns a list of ones as the gradient of the envelope wrt a step
        is 1 for that pixel and 0 everywhere else.

        Args:
            times: Array of times.

        Returns:
            PWC signal gradients.
        """
        gradient: Array = PWCGenerator._compute_gradient(
            self._multiply_flat_top,
            self._is_optimized(self._inphase),
            self._is_optimized(self._outofphase),
            self._time_grid,
            self._t_final,
            self._compute_pixel_index(times),
        )
        return gradient
