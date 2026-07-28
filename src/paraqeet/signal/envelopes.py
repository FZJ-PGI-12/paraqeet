"""Signal envelope shapes (constant, Gaussian, flat-top Gaussian, dCRAB) for pulse parametrization."""

import time
from functools import partial
from typing import override

import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.special import erf

from paraqeet.quantity import Array, Quantity
from paraqeet.signal.signal import Signal

jax.config.update("jax_enable_x64", True)


class Envelope(Signal):
    """Classical Signal Envelope class."""

    _amplitude: Quantity
    _t_final: Quantity

    def __init__(
        self,
        amplitude: Quantity | None = None,
        t_final: Quantity | None = None,
    ) -> None:
        """Initialize the envelope.

        Args:
            amplitude: The amplitude of the envelope.
            t_final: The length in time of the envelope.
        """
        self._amplitude = amplitude or Quantity(
            1.55e8,
            min_value=jnp.array(0.0),
            max_value=jnp.array(1e9),
            unit="Hz",
            name="Amplitude",
            two_pi=True,
        )

        self._t_final = t_final or Quantity(
            32e-9,
            min_value=jnp.array(0),
            max_value=jnp.array(100e-9),
            unit="s",
            name="t_final",
        )

    @override
    def get_parameters(self) -> list[Quantity]:
        return [self._amplitude, self._t_final]

    @property
    def amplitude(self) -> Quantity:
        """Return the amplitude of the envelope."""
        return self._amplitude

    @amplitude.setter
    def amplitude(self, amplitude: Quantity) -> None:
        """Set the amplitude of the envelope.

        Args:
            amplitude: Amplitude value of the system to be set.

        """
        self._amplitude = amplitude

    @property
    def t_final(self) -> Quantity:
        """Return the length of the envelope."""
        return self._t_final

    @t_final.setter
    def t_final(self, t_final: Quantity) -> None:
        """Set the length of the tone.

        Args:
            t_final: Length in time of the tone to be set.

        """
        self._t_final = t_final


class ConstantEnvelope(Envelope):
    """A constant envelope tone with a fixed length.

    The envelope has the value of the amplitude for all times up to ``t_final``
    and is zero afterwards.
    """

    @override
    @partial(jit, static_argnums=(0,))
    def _evaluate(
        self,
        amp: Array,
        t_final: Array,
        times: Array | float,
    ) -> Array:
        """Evaluate the envelope depending on all parameters.

        Args:
            amp: Amplitude.
            t_final: Final time.
            times: Array of times.
        """
        return jnp.squeeze(jnp.where(times <= t_final, amp, 0.0))

    @override
    def get_value(self, times: Array) -> Array:
        """Compute the constant signal envelope at different times.

        Args:
            times: Array of times.

        Returns:
            The constant signal at all times.
        """
        amp = self.amplitude.get_value()
        t_final = self.t_final.get_value()
        return self._evaluate(amp, t_final, times)  # type: ignore

    @override
    def get_time_gradient(self, times: Array) -> Array:
        """Compute a signal envelope's time derivative.

        Args:
            times: Array of times.

        Returns:
            An array of the signal's time derivative.
        """
        return jnp.zeros_like(times)


class ZeroEnvelope(ConstantEnvelope):
    """Shorthand implementation of a zero signal envelope."""

    def __init__(self) -> None:
        super().__init__()
        self.amplitude.set_value(0.0)


class FlatTopGaussianEnvelope(Envelope):
    """A flat-top Gaussian envelope."""

    _amplitude: Quantity
    _t_final: Quantity
    _t_up: Quantity
    _t_down: Quantity
    _ramp_time: Quantity

    def __init__(
        self,
        amplitude: Quantity | None = None,
        t_up: Quantity | None = None,
        t_down: Quantity | None = None,
        ramp_time: Quantity | None = None,
        t_final: Quantity | None = None,
    ) -> None:
        """Initialize the flat-top Gaussian envelope (a constant section framed by error-function shaped ramps).

        Args:
            amplitude: The amplitude of the envelope.
            t_up: The start time of constant section of the envelope.
            t_down: The end time of constant section of the envelope.
            ramp_time: The rate of ramp up and ramp down of the envelope.
            t_final: The length in time of the envelope. Used only if any of ``t_up``, ``t_down``,
                and ``ramp_time`` are none.
        """
        self._amplitude = amplitude or Quantity(
            1.55e8,
            min_value=jnp.array(0.0),
            max_value=jnp.array(1e9),
            unit="Hz",
            name="Amplitude",
            two_pi=True,
        )

        self._t_final = t_final or Quantity(
            32e-9,
            min_value=jnp.array(0),
            max_value=jnp.array(100e-9),
            unit="s",
            name="t_final",
        )

        self._t_up = t_up or Quantity(
            self._t_final.get_value() / 5,
            min_value=self._t_final.get_min_value(),
            max_value=self._t_final.get_max_value(),
            unit="s",
            name="t_up",
        )

        self._t_down = t_down or Quantity(
            4 * self._t_final.get_value() / 5,
            min_value=self._t_final.get_min_value(),
            max_value=self._t_final.get_max_value(),
            unit="s",
            name="t_down",
        )

        self._ramp_time = ramp_time or Quantity(
            self._t_final.get_value() / 10,
            min_value=self._t_final.get_min_value(),
            max_value=self._t_final.get_max_value(),
            unit="s",
            name="ramp_time",
        )

    @override
    def get_parameters(self) -> list[Quantity]:
        return [self._amplitude, self._t_up, self._t_down, self._ramp_time]

    @override
    @partial(jit, static_argnums=(0,))
    def _evaluate(self, amp: Array, t_up: Array, t_down: Array, ramp_time: Array, times: Array) -> Array:
        """Evaluate the flat-top Gaussian envelope.

        Explicitly depends on the optimizable parameters.

        Args:
            amp: Cosine pulse amplitude.
            t_up: The start time of constant section of the envelope.
            t_down: The end time of constant section of the envelope.
            ramp_time: The rate of ramp up and ramp down of the envelope.
            times: Array of times.

        Returns:
            The output of the device that explicitly depends
            on the optimizable parameters.

        """
        ramp_up = 1 + erf((times - t_up) / ramp_time)
        ramp_down = 1 + erf((-times + t_down) / ramp_time)
        return amp * ramp_up * ramp_down / 4

    @staticmethod
    @jit
    def _dir_erf(x: Array) -> Array:
        return 2 / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2))

    @partial(jit, static_argnums=(0,))
    def _evaluate_time_grad(self, amp: Array, t_up: Array, t_down: Array, ramp_time: Array, times: Array) -> Array:
        """Evaluate the time derivative of the envelope.

        Explicitly depends on the optimizable parameters.

        Args:
            amp: Cosine pulse amplitude.
            t_up: The start time of constant section of the envelope.
            t_down: The end time of constant section of the envelope.
            ramp_time: The rate of ramp up and ramp down of the envelope.
            times: Array of times.

        Returns:
            The time derivative of the envelope at the given timestamps.

        """
        ramp_up = 1 + erf((times - t_up) / ramp_time)
        ramp_up_t_dir = FlatTopGaussianEnvelope._dir_erf((times - t_up) / ramp_time)
        ramp_up_t_dir /= ramp_time

        ramp_down = 1 + erf((-times + t_down) / ramp_time)
        ramp_down_t_dir = FlatTopGaussianEnvelope._dir_erf((-times + t_down) / ramp_time)
        ramp_down_t_dir *= -1 / ramp_time

        prod_dir = ramp_up * ramp_down_t_dir + ramp_up_t_dir * ramp_down

        grad: Array = amp * prod_dir / 4
        return grad

    @partial(jit, static_argnums=(0,))
    def _evaluate_t_up_grad(self, amp: Array, t_up: Array, t_down: Array, ramp_time: Array, times: Array) -> Array:
        """Evaluate the gradient of the envelope with respect to ``t_up``.

        Explicitly depends on the optimizable parameters.

        Args:
            amp: Cosine pulse amplitude.
            t_up: The start time of constant section of the envelope.
            t_down: The end time of constant section of the envelope.
            ramp_time: The rate of ramp up and ramp down of the envelope.
            times: Array of times.

        Returns:
            The gradient of the envelope with respect to ``t_up``.

        """
        ramp_up_dir = FlatTopGaussianEnvelope._dir_erf((times - t_up) / ramp_time)
        ramp_up_dir /= -ramp_time
        ramp_down = 1 + erf((-times + t_down) / ramp_time)
        grad: Array = amp * ramp_up_dir * ramp_down / 4
        return grad

    @partial(jit, static_argnums=(0,))
    def _evaluate_t_down_grad(self, amp: Array, t_up: Array, t_down: Array, ramp_time: Array, times: Array) -> Array:
        """Evaluate the gradient of the envelope with respect to ``t_down``.

        Explicitly depends on the optimizable parameters.

        Args:
            amp: Cosine pulse amplitude.
            t_up: The start time of constant section of the envelope.
            t_down: The end time of constant section of the envelope.
            ramp_time: The rate of ramp up and ramp down of the envelope.
            times: Array of times.

        Returns:
            The gradient of the envelope with respect to ``t_down``.

        """
        ramp_up = 1 + erf((times - t_up) / ramp_time)
        ramp_down_dir = FlatTopGaussianEnvelope._dir_erf((-times + t_down) / ramp_time)
        ramp_down_dir /= ramp_time
        grad: Array = amp * ramp_up * ramp_down_dir / 4
        return grad

    @partial(jit, static_argnums=(0,))
    def _evaluate_ramp_time_grad(self, amp: Array, t_up: Array, t_down: Array, ramp_time: Array, times: Array) -> Array:
        """Evaluate the gradient of the envelope with respect to ``ramp_time``.

        Explicitly depends on the optimizable parameters.

        Args:
            amp: Cosine pulse amplitude.
            t_up: The start time of constant section of the envelope.
            t_down: The end time of constant section of the envelope.
            ramp_time: The rate of ramp up and ramp down of the envelope.
            times: Array of times.

        Returns:
            The gradient of the envelope with respect to ``ramp_time``.

        """
        ramp_up = 1 + erf((times - t_up) / ramp_time)
        ramp_up_dir = FlatTopGaussianEnvelope._dir_erf((times - t_up) / ramp_time)
        ramp_up_dir *= -(times - t_up) / (ramp_time**2)

        ramp_down = 1 + erf((-times + t_down) / ramp_time)
        ramp_down_dir = FlatTopGaussianEnvelope._dir_erf((-times + t_down) / ramp_time)
        ramp_down_dir *= -(-times + t_down) / (ramp_time**2)

        prod_dir = ramp_up * ramp_down_dir + ramp_up_dir * ramp_down

        grad: Array = amp * prod_dir / 4
        return grad

    @override
    def get_value(self, times: Array) -> Array:
        """Compute the flat-top Gaussian envelope at the given timestamps.

        Args:
            times: Array of times.

        Returns:
            The envelope evaluated at the given timestamps.

        """
        amp = self._amplitude.get_value()
        t_up = self._t_up.get_value()
        t_down = self._t_down.get_value()
        ramp_time = self._ramp_time.get_value()
        # returns JitWrapped
        return self._evaluate(amp, t_up, t_down, ramp_time, times)  # type: ignore

    @override
    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
        """Return the gradient wrt dimensionless parameters.

        Args:
            times: Array of times.

        Returns:
            The gradient wrt dimensionless parameters.

        """
        amp = self._amplitude.get_value()
        t_up = self._t_up.get_value()
        t_down = self._t_down.get_value()
        ramp_time = self._ramp_time.get_value()
        t_arr = jnp.array(times, ndmin=1)

        grads = []
        if self._is_optimized(self.amplitude):
            grads.append(self._evaluate(jnp.array([1.0]), t_up, t_down, ramp_time, t_arr))
        if self._is_optimized(self._t_up):
            grads.append(self._evaluate_t_up_grad(amp, t_up, t_down, ramp_time, t_arr))
        if self._is_optimized(self._t_down):
            grads.append(self._evaluate_t_down_grad(amp, t_up, t_down, ramp_time, t_arr))
        if self._is_optimized(self._ramp_time):
            grads.append(self._evaluate_ramp_time_grad(amp, t_up, t_down, ramp_time, t_arr))

        gradient = jnp.stack(grads, axis=1) if len(grads) > 0 else jnp.empty((t_arr.shape[0], 0))
        return self._evaluate(amp, t_up, t_down, ramp_time, times), gradient

    @override
    def get_time_gradient(self, times: Array) -> Array:
        """Compute a signal envelope's time derivative.

        Args:
            times: Array of times.

        Returns:
            A vector of the signal's time derivative.
        """
        amp = self.amplitude.get_value()
        t_up = self._t_up.get_value()
        t_down = self._t_down.get_value()
        ramp_time = self._ramp_time.get_value()
        return jnp.array(self._evaluate_time_grad(amp, t_up, t_down, ramp_time, times))


class GaussEnvelope(Envelope):
    """A simple Gaussian envelope.

    The Gaussian is centered at ``t_final / 2`` with a standard deviation of
    ``t_final / 8``.

    Attributes:
        _amplitude: The amplitude of the envelope.
        _t_final: The length in time of the envelope.

    """

    @override
    @partial(jax.jit, static_argnums=(0,))
    def _evaluate(self, amp: Array, t_final: Array, times: Array) -> Array:  # type: ignore
        """Calculate the Gaussian signal.

        Args:
            t_final: Duration of the signal to calculate the center of the Gaussian from.
            times: Array of times.

        Returns:
            The Gaussian signal.
        """
        sigma = t_final / 8
        env = amp * jnp.exp(-(1 / 2) * (times - t_final / 2) ** 2 / sigma**2)
        return jnp.squeeze(env)

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate_time_gradient(self, amp: Array, t_final: Array, times: Array) -> Array:
        """Calculate the time derivative of the Gaussian signal.

        Args:
            amp: Amplitude of the Gaussian.
            t_final: Duration of the signal to calculate the center of the Gaussian from.
            times: Array of times.

        Returns:
            The Gaussian signal's time derivative.
        """
        sigma = t_final / 8
        time_grad = self._evaluate(amp, t_final, times) * -1.0 * (times - t_final / 2) / sigma**2
        # returns JitWrapped
        return time_grad  # type: ignore

    @override
    def get_time_gradient(self, times: Array) -> Array:
        """Compute a signal envelope's time derivative.

        Args:
            times: Array of times.

        Returns:
            A vector of the signal's time derivative.
        """
        t_final = self.t_final.get_value()
        amp = self.amplitude.get_value()
        env_time_deriv = self._evaluate_time_gradient(amp, t_final, times)
        # returns JitWrapped
        return env_time_deriv  # type: ignore


class DCRABEnvelope(Envelope):
    r"""Create a dCRAB pulse envelope.

    The dCRAB pulse is given as a sum of sinusoidal components as :cite:p:`rach2015dressing,muller2022one`

    .. math::
        f(t) = g(t)  \left( 1 + \sum_{i=1}^{N_c / 2} c_{2i} \frac{\cos(\omega_{2i} t)}{\Lambda(t)}
        + \sum_{i = 1} ^ {N_c/2} c_{2i + 1} \frac{\sin(\omega_{2i + 1} t)}{\Lambda(t)} \right)

    Here we consider :math:`g(t) = \Lambda(t) = 1` for simplicity.
    Further, even components are for cosine and odd components are for sine.

    Note:
        The function is designed to work well for an even total number of components.
        For an odd total number, it may not work as expected.

    Attributes:
        _total_num_components: Total number of components in the current dCRAB basis. This is the number of coefficients
            or the number of frequencies present. NOT the sum of them.
        _real_coefficients: Vector quantity as a list of amplitudes of individual sinusoidal components.
        _real_frequencies: Vector quantity as a list of frequencies of individual sinusoidal components.
        _real_phases: Vector quantity as a list of phases of individual sinusoidal components.
    """

    _amplitude: Quantity
    _t_final: Quantity
    _num_components: int
    _total_num_components: int
    _real_coefficients: list[Quantity]
    _real_frequencies: list[Quantity]
    _real_phases: list[Quantity]
    _imag_coefficients: list[Quantity]
    _imag_frequencies: list[Quantity]
    _imag_phases: list[Quantity]
    _min_frequency: float
    _max_frequency: float

    def __init__(
        self,
        amplitude: Quantity | None = None,
        t_final: Quantity | None = None,
        num_components: int = 2,
        min_frequency: float = 0.0,
        max_frequency: float = 2 * jnp.pi * 5.0,
        seed: int | None = None,
    ) -> None:
        """Initialize the dCRAB pulse envelope.

        Args:
            amplitude: The amplitude of the envelope.
            t_final: The length in time of the envelope.
            num_components: Number of components added each iteration to the dCRAB basis. Defaults to 2.
                Advised to be an even number.
            min_frequency: Minimum frequency of the randomized sinusoidal components.
            max_frequency: Maximum frequency of the randomized sinusoidal components.
            seed: Seed for the randomized coefficients, frequencies, and phases.
                If None, a seed is derived from the current time.
        """
        self._amplitude = amplitude or Quantity(
            1.55e8,
            min_value=jnp.array(0.0),
            max_value=jnp.array(1e9),
            unit="Hz",
            name="Amplitude",
            two_pi=True,
        )

        self._t_final = t_final or Quantity(
            32e-9,
            min_value=jnp.array(0),
            max_value=jnp.array(100e-9),
            unit="s",
            name="t_final",
        )
        self._min_frequency = min_frequency
        self._max_frequency = max_frequency

        self._num_components = num_components

        if seed is None:
            seed = int(1e7 * time.time())

        key = jax.random.key(seed)
        coeffs = jax.random.uniform(key, shape=(self._num_components,), minval=-1, maxval=1)

        self._real_coefficients = [
            Quantity(
                coeffs[i],
                min_value=jnp.array(-1.0),
                max_value=jnp.array(1.0),
                unit="",
                name=f"CRAB Re coefficient {i}",
            )
            for i in range(self._num_components)
        ]

        key = jax.random.key(2 * seed + 91)
        coeffs = jax.random.uniform(key, shape=(self._num_components,), minval=-1, maxval=1)

        self._imag_coefficients = [
            Quantity(
                coeffs[i],
                min_value=jnp.array(-1.0),
                max_value=jnp.array(1.0),
                unit="",
                name=f"CRAB Im coefficient {i}",
            )
            for i in range(self._num_components)
        ]

        key = jax.random.key(3 * seed + 81)
        freqs = jax.random.uniform(
            key, shape=(self._num_components,), minval=self._min_frequency, maxval=self._max_frequency
        )

        self._real_frequencies = [
            Quantity(
                freqs[i],
                min_value=jnp.array(self._min_frequency),
                max_value=jnp.array(self._max_frequency),
                unit="Hz",
                name=f"CRAB Re frequency {i}",
                two_pi=True,
            )
            for i in range(self._num_components)
        ]

        key = jax.random.key(4 * seed + 19)
        freqs = jax.random.uniform(
            key, shape=(self._num_components,), minval=self._min_frequency, maxval=self._max_frequency
        )

        self._imag_frequencies = [
            Quantity(
                freqs[i],
                min_value=jnp.array(self._min_frequency),
                max_value=jnp.array(self._max_frequency),
                unit="Hz",
                name=f"CRAB Im frequency {i}",
                two_pi=True,
            )
            for i in range(self._num_components)
        ]

        key = jax.random.key(5 * seed + 73)
        phases = jax.random.uniform(key, shape=(self._num_components,), minval=-jnp.pi, maxval=jnp.pi)

        self._real_phases = [
            Quantity(
                phases[i],
                min_value=jnp.array(-jnp.pi),
                max_value=jnp.array(jnp.pi),
                unit="rad",
                name=f"CRAB Re Phase {i}",
            )
            for i in range(self._num_components)
        ]

        key = jax.random.key(6 * seed + 74)
        phases = jax.random.uniform(key, shape=(self._num_components,), minval=-jnp.pi, maxval=jnp.pi)

        self._imag_phases = [
            Quantity(
                phases[i],
                min_value=jnp.array(-jnp.pi),
                max_value=jnp.array(jnp.pi),
                unit="rad",
                name=f"CRAB Im Phase {i}",
            )
            for i in range(self._num_components)
        ]

        self._total_num_components = self._num_components

    @override
    def get_parameters(self) -> list[Quantity]:
        """Return the parameters of the CRAB signal.

        The parameters are arranged as follows,
        ``[amplitude, t_final, real coefficients, real frequencies, real phases,
        imag coefficients, imag frequencies, imag phases]``

        """
        params = [self.amplitude, self._t_final]
        params.extend(self._real_coefficients)
        params.extend(self._real_frequencies)
        params.extend(self._real_phases)
        params.extend(self._imag_coefficients)
        params.extend(self._imag_frequencies)
        params.extend(self._imag_phases)
        return params

    def get_coefficients_frequencies_and_phases(self) -> list[Quantity]:
        """Return all the coefficients, frequencies and phases used in the CRAB signal."""
        return (
            self._real_coefficients
            + self._real_frequencies
            + self._real_phases
            + self._imag_coefficients
            + self._imag_frequencies
            + self._imag_phases
        )

    def add_new_components(self, seed: int | None = None) -> None:
        """Add ``self._num_components`` number of new randomized components to the optimization."""
        if seed is None:
            seed = int(1e7 * time.time())

        # Limit max coeff value so that the new components do not derail the optimization.
        key = jax.random.key(seed)
        coeffs = jax.random.uniform(key, shape=(self._num_components,), minval=-0.5, maxval=0.5)

        self._real_coefficients.extend(
            [
                Quantity(
                    coeffs[i],
                    min_value=jnp.array(-1.0),
                    max_value=jnp.array(1.0),
                    unit="",
                    name=f"CRAB Re coefficient {i + self._total_num_components}",
                )
                for i in range(self._num_components)
            ]
        )

        key = jax.random.key(4 * seed + 59)
        freqs = jax.random.uniform(
            key, shape=(self._num_components,), minval=self._min_frequency, maxval=self._max_frequency
        )

        self._real_frequencies.extend(
            [
                Quantity(
                    freqs[i],
                    min_value=jnp.array(self._min_frequency),
                    max_value=jnp.array(self._max_frequency),
                    unit="Hz",
                    name=f"CRAB Re frequency {i + self._total_num_components}",
                    two_pi=True,
                )
                for i in range(self._num_components)
            ]
        )

        key = jax.random.key(8 * seed + 61)
        phases = jax.random.uniform(key, shape=(self._num_components,), minval=-jnp.pi, maxval=jnp.pi)

        self._real_phases.extend(
            [
                Quantity(
                    phases[i],
                    min_value=jnp.array(-jnp.pi),
                    max_value=jnp.array(jnp.pi),
                    unit="rad",
                    name=f"CRAB Re Phase {i + self._total_num_components}",
                )
                for i in range(self._num_components)
            ]
        )

        key = jax.random.key(10 * seed)
        coeffs = jax.random.uniform(key, shape=(self._num_components,), minval=-0.5, maxval=0.5)

        self._imag_coefficients.extend(
            [
                Quantity(
                    coeffs[i],
                    min_value=jnp.array(-1.0),
                    max_value=jnp.array(1.0),
                    unit="",
                    name=f"CRAB Im coefficient {i + self._total_num_components}",
                )
                for i in range(self._num_components)
            ]
        )

        key = jax.random.key(7 * seed + 89)
        freqs = jax.random.uniform(
            key, shape=(self._num_components,), minval=self._min_frequency, maxval=self._max_frequency
        )

        self._imag_frequencies.extend(
            [
                Quantity(
                    freqs[i],
                    min_value=jnp.array(self._min_frequency),
                    max_value=jnp.array(self._max_frequency),
                    unit="Hz",
                    name=f"CRAB Im frequency {i + self._total_num_components}",
                    two_pi=True,
                )
                for i in range(self._num_components)
            ]
        )

        key = jax.random.key(21 * seed + 34)
        phases = jax.random.uniform(key, shape=(self._num_components,), minval=-jnp.pi, maxval=jnp.pi)

        self._imag_phases.extend(
            [
                Quantity(
                    phases[i],
                    min_value=jnp.array(-jnp.pi),
                    max_value=jnp.array(jnp.pi),
                    unit="rad",
                    name=f"CRAB Im phase {i + self._total_num_components}",
                )
                for i in range(self._num_components)
            ]
        )

        self._total_num_components += self._num_components

    def remove_small_coefficients(self, tol: float = 1e-7) -> None:
        """Remove coefficients (and corresponding frequencies) that are smaller than a tolerance."""
        removed_coeffs = []
        for i, real_coeff, imag_coeff in zip(
            range(self._total_num_components), self._real_coefficients, self._imag_coefficients
        ):
            if jnp.abs(real_coeff.get_value() + 1j * imag_coeff.get_value()) < tol:
                removed_coeffs.append(i)

        self._real_coefficients = [coeff for i, coeff in enumerate(self._real_coefficients) if i not in removed_coeffs]
        self._real_frequencies = [freq for i, freq in enumerate(self._real_frequencies) if i not in removed_coeffs]
        self._real_phases = [phase for i, phase in enumerate(self._real_phases) if i not in removed_coeffs]

        self._imag_coefficients = [coeff for i, coeff in enumerate(self._imag_coefficients) if i not in removed_coeffs]
        self._imag_frequencies = [freq for i, freq in enumerate(self._imag_frequencies) if i not in removed_coeffs]
        self._imag_phases = [phase for i, phase in enumerate(self._imag_phases) if i not in removed_coeffs]

        self._total_num_components = len(self._real_coefficients)

    @override
    @partial(jax.jit, static_argnums=(0,))
    def _evaluate(self, *params: Array) -> Array:  # type: ignore
        """Compute the CRAB pulse.

        Here the params are arranged as follows:
            [amplitude, t_final, ... total_num coefficients ..., ... total_num frequencies ..., t]

        This evaluate function is written in this way to make it compatible with adding new
        components and freezing existing components required for dCRAB optimization.
        """
        amp = params[0]
        t_final = params[1]
        real_coeffs: list[Array] = params[2 : 2 + self._total_num_components]  # type: ignore
        real_freqs: list[Array] = params[2 + self._total_num_components : 2 + 2 * self._total_num_components]  # type: ignore
        real_phases: list[Array] = params[2 + 2 * self._total_num_components : 2 + 3 * self._total_num_components]  # type: ignore

        imag_coeffs: list[Array] = params[2 + 3 * self._total_num_components : 2 + 4 * self._total_num_components]  # type: ignore
        imag_freqs: list[Array] = params[2 + 4 * self._total_num_components : 2 + 5 * self._total_num_components]  # type: ignore
        imag_phases: list[Array] = params[2 + 5 * self._total_num_components : 2 + 6 * self._total_num_components]  # type: ignore

        times = params[-1]
        env_real = jnp.zeros_like(times)
        env_imag = jnp.zeros_like(times)
        for i in range(self._total_num_components):
            env_real += real_coeffs[i] * jnp.cos(real_freqs[i] * times / t_final + real_phases[i])
            env_imag += imag_coeffs[i] * jnp.cos(imag_freqs[i] * times / t_final + imag_phases[i])
        env_real /= 2 * jnp.sum(jnp.abs(jnp.array(real_coeffs)))
        env_imag /= 2 * jnp.sum(jnp.abs(jnp.array(imag_coeffs)))

        env = env_real + 1j * env_imag
        return jnp.squeeze(amp * env)

    @override
    def get_value(self, times: Array) -> Array:
        t_final = self.t_final.get_value()
        amp = self.amplitude.get_value()
        real_coeffs = [coeff.get_value() for coeff in self._real_coefficients]
        real_freqs = [freq.get_value() for freq in self._real_frequencies]
        real_phases = [phase.get_value() for phase in self._real_phases]

        imag_coeffs = [coeff.get_value() for coeff in self._imag_coefficients]
        imag_freqs = [freq.get_value() for freq in self._imag_frequencies]
        imag_phases = [phase.get_value() for phase in self._imag_phases]

        params = [amp, t_final] + real_coeffs + real_freqs + real_phases + imag_coeffs + imag_freqs + imag_phases
        return self._evaluate(*params, times)  # type: ignore
