"""Class definition for the Sinusoidal generator model."""

import jax.numpy as jnp

from paraqeet.quantity import Array, Quantity
from paraqeet.signal.generator import Generator
from paraqeet.signal.waveform import LocalOscillator, Waveform


class IQMixer(Generator):
    """Control signal generation.

    Waveforms of envelopes (low bandwith) are mixed with a local oscillator
    (high bandwidth) to apply a desired control field to the system.

    Parameters
    ----------
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

    def get_parameters(self) -> list[Quantity]:
        """Return a list of parameters.

        Collects and returns a list of parameters from the tone, generator
        and the carrier signal.

        Returns
        -------
        list[Quantity]
            All Parameters describing the signal.
        """
        pars = []
        for env in self._envs:
            pars += env.get_parameters()
        pars += self._lo.get_parameters()
        pars += [self._phase]
        return pars

    def set_optimizable_parameters(self, params: list[Quantity]) -> None:
        """Set specified parameters to be optimized.

        Parameters
        ----------
        params : list[Quantity]
        """
        super().set_optimizable_parameters(params)

        for dev in self._envs:
            dev.set_optimizable_parameters(params)

        self._lo.set_optimizable_parameters(params)

    def _complex_signal(self, times: Array) -> Array:
        """Generate a signal for time(s).

        Doesnt take real value now for ease of gradient computation.

        Parameters
        ----------
        times: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the signal vector.

        """
        env = jnp.zeros_like(times)
        for dev in self._envs:
            env += jnp.reshape(dev.get_value(times), env.shape)
        sig = env.conj() * self._lo.get_value(times)
        sig = sig * jnp.exp(-1j * self._phase.get_value())
        return sig

    def get_value(self, times: Array) -> Array:
        """Generate a signal for time(s).

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the signal vector.

        """
        return jnp.real(self._complex_signal(times))

    def get_value_and_gradient(self, times) -> tuple[Array, Array]:
        r"""Collect and returns the gradients from all devices.

        Since the

        .. math::
            signal = \\Re(\\epsilon(t)^*  \\exp(i \\omega t)  \\exp(-i \\phi))

        Derivative of the signal wrt optimizable parameter of envelope would be

        .. math::
            0.5 * \\Re(\\partial \\epsilon(t)^* \\exp(i \\omega t)  \\exp(-i \\phi))

        (TODO - Check the envelope derivatives)

        And derivative of signal wrt parameter of LO would be

        .. math::
            0.5 i t \\epsilon(t)^* \\exp(i \\omega t)  \\exp(-i \\phi))

        And derivative of signal wrt phase would be

        .. math::
            -0.5 i \\epsilon(t)^* \\exp(i \\omega t)  \\exp(-i \\phi))

        The 0.5 are due to the Wirtinger derivatives due to Re part.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the signal gradient vector.

        """
        phase_fac = jnp.exp(-1j * self._phase.get_value())
        lo_out = self._lo.get_value(times)
        sig = self._complex_signal(times)
        gradients = jnp.zeros(shape=(times.shape[0], 0))

        # Collect gradients for envelopes
        for dev in self._envs:
            _, grad = dev.get_value_and_gradient(times)
            grad = grad.conj()
            if grad.size != 0:
                grad *= jnp.expand_dims(lo_out * phase_fac, axis=1)
            gradients = jnp.append(gradients, 0.5 * jnp.real(grad), axis=1)

        # Collect LO gradients
        lo_freq = self._lo.get_parameters()[0]
        if self._is_optimized(lo_freq):
            gradients = jnp.append(
                gradients,
                jnp.expand_dims(0.5j * times * sig, 1),
                axis=1,
            )

        # Collect gradient of Phase
        if self._is_optimized(self._phase):
            gradients = jnp.append(
                gradients,
                jnp.expand_dims(-0.5j * sig, 1),
                axis=1,
            )
        return jnp.real(sig), gradients

    def get_gradient_at_timestep(self, time: Array) -> Array:
        r"""Return the gradients from all devices at the given time.

        Since the

        .. math::
            signal = \\Re(\\epsilon(t)^*  \\exp(i \\omega t)  \\exp(-i \\phi))

        Derivative of the signal wrt optimizable parameter of envelope would be

        .. math::
            0.5 * \\Re(\\partial \\epsilon(t)^* \\exp(i \\omega t)  \\exp(-i \\phi))

        (TODO - Check the envelope derivatives)

        And derivative of signal wrt parameter of LO would be

        .. math::
            0.5 i t \\epsilon(t)^* \\exp(i \\omega t)  \\exp(-i \\phi))

        And derivative of signal wrt phase would be

        .. math::
            -0.5 i \\epsilon(t)^* \\exp(i \\omega t)  \\exp(-i \\phi))

        The 0.5 are due to the Wirtinger derivatives due to Re part.

        Parameters
        ----------
        time: Array
            Single timestamp.

        Returns
        -------
        Array
            Return the gradients from all devices at one time.

        """
        phase_fac = jnp.exp(-1j * self._phase.get_value())
        lo_out = jnp.squeeze(self._lo.get_value(time), axis=0)
        sig = self._complex_signal(time)
        gradients = jnp.zeros(shape=(0,))

        # Collect gradients for envelopes
        for dev in self._envs:
            _, grad = dev.get_value_and_gradient(time)
            grad = jnp.squeeze(grad.conj(), axis=0)
            if grad.size != 0:
                grad *= lo_out * phase_fac
            gradients = jnp.append(gradients, 0.5 * jnp.real(grad), axis=0)

        # Collect LO gradients
        lo_freq = self._lo.get_parameters()[0]
        if self._is_optimized(lo_freq):
            gradients = jnp.append(gradients, 0.5j * time * sig, axis=0)

        # Collect gradient of Phase
        if self._is_optimized(self._phase):
            gradients = jnp.append(
                gradients,
                -0.5j * sig,
                axis=0,
            )
        return gradients
