"""Class definition for the Sinusoidal generator model."""

from typing import override

import jax.numpy as jnp

from paraqeet.quantity import Array, Quantity
from paraqeet.signal.generator import Generator
from paraqeet.signal.waveform import LocalOscillator, Waveform


class ComplexIQMixer(Generator):
    """Control signal generation.

    Waveforms of envelopes (low bandwidth) are mixed with a local oscillator
    (high bandwidth) to apply a desired complex control field to the system.

    Args:
        envelopes : List[Waveform]
            List of input devices.
        frequency : Quantity | None
            Frequency of local oscillator.
        phase : Quantity | None
            Phase of local oscillator.
    """

    _envs: list[Waveform]
    _phase: Quantity
    _optimizable_parameters: list[Quantity] = []

    def __init__(
        self,
        envelopes: list[Waveform] | None,
        frequency: Quantity | None = None,
        phase: Quantity | None = None,
    ):
        self._envs = envelopes or []

        self._lo = LocalOscillator(frequency=frequency)

        self._phase = phase or Quantity(
            jnp.array(0.0),
            min_value=jnp.array(-jnp.pi),
            max_value=jnp.array(jnp.pi),
            unit="rad",
            name="Phase",
        )

    @override
    def get_parameters(self) -> list[Quantity]:
        """Return a list of parameters.

        Collects and returns a list of parameters from the tone, generator
        and the carrier signal.

        Returns:
            list[Quantity]
                All Parameters describing the signal.
        """
        pars = []
        for env in self._envs:
            pars += env.get_parameters()
        pars += self._lo.get_parameters()
        pars += [self._phase]
        return pars

    @override
    def set_optimizable_parameters(self, params: list[Quantity]) -> None:
        """Set specified parameters to be optimized.

        Args:
            params : list[Quantity]
        """
        super().set_optimizable_parameters(params)

        for dev in self._envs:
            dev.set_optimizable_parameters(params)

        self._lo.set_optimizable_parameters(params)

    def _complex_signal(self, times: Array) -> Array:
        """Generate a signal for time(s).

        Doesn't take real value now for ease of gradient computation.

        Args:
            times: Array
                One-dimensional vector of timestamps.

        Returns:
            Array
                Returns the signal vector.

        """
        env = jnp.zeros_like(times)
        for dev in self._envs:
            env += jnp.reshape(dev.get_value(times), env.shape)
        sig = env.conj() * self._lo.get_value(times)
        sig = sig * jnp.exp(-1j * self._phase.get_value())
        return sig

    @override
    def get_value(self, times: Array) -> Array:
        """Generate a signal for time(s).

        Args:
            times: Array
                One-dimensional vector of timestamps.

        Returns:
            Array
                Returns the signal vector.

        """
        return self._complex_signal(times)

    def _complex_signal_and_gradient(self, times: Array) -> tuple[Array, Array]:
        r"""Collect and returns the gradients from all devices.

        Since the

        .. math::
            signal = \epsilon(t)^*  \exp(i \omega t)  \exp(-i \phi)

        The derivative of the signal with respect to a real parameter p is

        .. math::
            \frac{\partial}{\partial p} z = \frac{\partial z}{\partial p}

        Args:
            times: Array
                One-dimensional vector of timestamps.

        Returns:
            Array
                The signal gradient vector as a (n_times, n_params) array.
        """
        phase_fac = jnp.exp(-1j * self._phase.get_value())
        lo_out = self._lo.get_value(times)
        sig = self._complex_signal(times)
        gradient = jnp.zeros(shape=(times.shape[0], 0))

        # Collect gradients for envelopes
        # d(Re(sig))/dp = Re(d(env.conj())/dp * lo_out * phase_fac)
        for dev in self._envs:
            _, grad = dev.get_value_and_gradient(times)
            grad = grad.conj()
            if grad.size != 0:
                grad *= jnp.expand_dims(lo_out * phase_fac, axis=1)
            gradient = jnp.append(gradient, grad, axis=1)

        # Collect LO gradients
        # d(Re(sig))/d(lo) = Re(1j * t * sig)
        lo_freq = self._lo.get_parameters()[0]
        if self._is_optimized(lo_freq):
            gradient = jnp.append(
                gradient,
                jnp.expand_dims(1j * times * sig, 1),
                axis=1,
            )

        # Collect gradient of Phase
        # d(Re(sig))/d(phase) = Re(-1j * sig)
        if self._is_optimized(self._phase):
            gradient = jnp.append(
                gradient,
                jnp.expand_dims(-1j * sig, 1),
                axis=1,
            )
        return sig, gradient

    @override
    def get_gradient(self, times: Array) -> Array:
        _, gradient = self.get_value_and_gradient(times)
        return gradient

    @override
    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
        return self._complex_signal_and_gradient(times)


class IQMixer(ComplexIQMixer):
    """Control signal generation.

    Waveforms of envelopes (low bandwidth) are mixed with a local oscillator
    (high bandwidth) to apply a desired real control field to the system.

    Args:
        envelopes : List[Waveform]
            List of input devices.
        frequency : Quantity | None
            Frequency of local oscillator.
        phase : Quantity | None
            Phase of local oscillator.
    """

    @override
    def get_value(self, times: Array) -> Array:
        return jnp.real(self._complex_signal(times))

    @override
    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
        sig, gradient = self._complex_signal_and_gradient(times)
        return jnp.real(sig), jnp.real(gradient)
