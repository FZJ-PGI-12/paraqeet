from functools import partial
from typing import override

import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import erf

from paraqeet.quantity import Array, Quantity
from paraqeet.signal.generator import Generator
from paraqeet.signal.waveform import Waveform


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

    _envs: list[Waveform]
        List of Envelopes
    _tlist: Array
        Left time points for discretization. These can be used for propagation and optimization.
    _time_grid: Array
        Time grid used to discretize the pulse. These are shifted from tlist by dt, and doesn't include zero time.
    _max_amplitude: float
        Maximum amplitude of the drive
    _inphase: Quantity
        The in-phase component of the pulse
    _outofphase: Quantity
        The out-of-phase component of the pulse
    _optimizable_paramters: list[Quantity]
        List of own parameters that would be optimized by the optimizer.
    _multiply_flat_top: bool
        Flag to multiply flat-top-Gaussain pulse to the signal to ensure it
        starts and ends at zero.

    Parameters
    ----------
    envelopes : List[Waveform]
        List of input devices.

    """

    _envs: list[Waveform]
    _tlist: Array
    _time_grid: Array
    _max_amplitude: float
    _inphase: Quantity
    _outofphase: Quantity
    _optimizable_parameters: list[Quantity] = []
    _multiply_flat_top: bool = False

    def __init__(
        self,
        envelopes: list[Waveform] | None,
        tlist: Array,
        max_amplitude: float | None = None,
    ):
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
    def _compute_envelope(self, t):
        t_final = self._t_final
        ramp_time = t_final / 25
        ramp_up = 1 + erf((t - 2 * t_final / 20) / ramp_time)
        ramp_down = 1 + erf((-t + 18 * t_final / 20) / ramp_time)
        return ramp_up * ramp_down / 4

    @property
    def tlist(self) -> Array:
        """Get time grid discritization for generating PWC pulse.

        Returns
        -------
        Array
            Array of time points at which envelope is discritized.
        """
        return self._tlist

    @tlist.setter
    def tlist(self, tlist: Array) -> None:
        """Set time grid discritization for generating PWC pulse.

        Parameters
        ----------
        tlist : Array
            Array of time points at which envelope is discritized.
        """
        self._tlist = tlist

        # update time grid
        dt = tlist[1] - tlist[0]
        self._time_grid = tlist[:-1] + dt / 2

        self._setup_inphase_and_outofphase()

    @property
    def max_amplitude(self) -> float:
        """Get the maximum drive amplitude.

        Returns
        -------
            The value of the maximum drive amplitude.
        """
        return self._max_amplitude

    @max_amplitude.setter
    def max_amplitude(self, max_amplitude: float) -> None:
        """Set the maximum drive amplitude of the drive.

        Parameters
        ----------
        max_amplitude: float
            The value of the maximu drive amplitude.
        """
        self._max_amplitude = max_amplitude
        self._setup_inphase_and_outofphase()

    @property
    def envs(self) -> list[Waveform]:
        """Gets the list of envelopes.

        Returns
        -------
        list[Waveform]
            The list of waveforms associated with the generator.
        """
        return self._envs

    @property
    def multiply_flat_top(self) -> bool:
        """Flag to multiply the pulse with a FlatTop.

        This can be used to make the start and end values zeros and force the
        PWC pulse to change smoothly.

        Returns
        -------
        multiply_flat_top: bool
            Flag value for multiply_flat_top.
        """
        return self._multiply_flat_top

    @multiply_flat_top.setter
    def multiply_flat_top(self, multiply_flat_top: bool) -> None:
        """Set flag to multiply the pulse with a FlatTop.

        This can be used to make the start and end values zeros and force the
        PWC pulse to change smoothly.

        Parameters
        ----------
        multiply_flat_top : bool
            Flag value for multiply_flat_top.
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

    def _get_partial_derivatives(self, times) -> Array:
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

        Returns
        -------
        list[Quantity]
            All Parameters describing the signal.
        """
        return [self._inphase, self._outofphase]

    @override
    def set_optimizable_parameters(self, params: list[Quantity]) -> None:
        """Set specified parameters to be optimized.

        Optimizable paramters can be inphase and out-of-phase.

        Parameters
        ----------
        params : list[Quantity]
        """
        super().set_optimizable_parameters(params)

    @partial(jit, static_argnums=(0,))
    def _pwc_signal(
        self,
        inphase: Array,
        outofphase: Array,
        dt: Array,
        t: Array,
    ) -> Array:
        """Generate a signal for a single time point 't'.

        The PWC signal is generated by finding the closest time point
        and returning the correspoinding amplitude value.

        Parameters
        ----------
        inphase: Array
            1-D vector of step values of real part of the PWC signal.
        out-of-phase: Array
            1-D vector of step values of complex part of the PWC signal.
        tlist: Array
            Time bins of the PWC pulse.
        t: Array
            One time point.

        Returns
        -------
        Array
            Returns the PWC signal value at t.

        """
        index = jnp.array(t / dt, int)
        return inphase[index] + 1.0j * outofphase[index]

    @override
    def get_value(self, times: Array) -> Array:
        """Generate the PWC signal for time(s) 't'.

        Parameters
        ----------
        times: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the signal vector.

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

        Parameters
        ----------
        times : Array
            Array of time steps.

        Returns
        -------
        Array
            PWC signal gradients.
        """
        grads = []
        time_grid = self._time_grid

        inphase = self._inphase.get_value()
        outofphase = self._outofphase.get_value()

        if self._multiply_flat_top:
            smoothing = self._compute_envelope(time_grid)
            index = jnp.argmin(jnp.abs(jnp.expand_dims(time_grid, axis=1) - times), axis=0)
            env = smoothing[index]
            inphase *= env
            outofphase *= env
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
