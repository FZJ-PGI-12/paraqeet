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

    """

    __envs: list[Waveform]
    __phase: Quantity
    _optimizable_parameters: list[Quantity] = []

    def __init__(
        self,
        envelopes: list[Waveform] | None,
        frequency: Quantity | None = None,
        phase: Quantity | None = None,
    ):
        self.__envs = envelopes or []

        self.__lo = LocalOscillator(frequency=frequency)

        self.__phase = phase or Quantity(
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
        for env in self.__envs:
            pars += env.get_parameters()
        pars += self.__lo.get_parameters()
        pars += [self.__phase]
        return pars

    def set_optimizable_parameters(self, params: list[Quantity]) -> None:
        """Set specified parameters to be optimized.

        Parameters
        ----------
        params : list[Quantity]
        """
        super().set_optimizable_parameters(params)

        for dev in self.__envs:
            dev.set_optimizable_parameters(params)

        self.__lo.set_optimizable_parameters(params)

    def __complex_signal(self, t: Array) -> Array:
        """Generate a signal for time(s) 't'.

        Doesnt take real value now for ease of gradient computation.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the signal vector.

        """
        env = jnp.zeros_like(t)
        for dev in self.__envs:
            env += jnp.reshape(dev.compute_output(t), env.shape)
        sig = env.conj() * self.__lo.compute_output(t)
        sig = sig * jnp.exp(-1j * self.__phase.get_value())
        return sig

    def generate_signal(self, times: Array) -> Array:
        """Generate a signal for time(s) 't'.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the signal vector.

        """
        return jnp.real(self.__complex_signal(times))

    def generate_signal_gradient(self, times) -> Array:
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
        phase_fac = jnp.exp(-1j * self.__phase.get_value())
        lo_out = self.__lo.compute_output(times)
        sig = self.__complex_signal(times)
        gradients = jnp.zeros(shape=(times.shape[0], 0))

        # Collect gradients for envelopes
        for dev in self.__envs:
            grad = dev.compute_gradient(times).conj()
            if grad.size != 0:
                grad *= jnp.expand_dims(lo_out * phase_fac, axis=1)
            gradients = jnp.append(gradients, 0.5 * jnp.real(grad), axis=1)

        # Collect LO gradients
        lo_freq = self.__lo.get_parameters()[0]
        if self._is_optimized(lo_freq):
            gradients = jnp.append(
                gradients,
                jnp.expand_dims(0.5j * times * sig, 1),
                axis=1,
            )

        # Collect gradient of Phase
        if self._is_optimized(self.__phase):
            gradients = jnp.append(
                gradients,
                jnp.expand_dims(-0.5j * sig, 1),
                axis=1,
            )
        return gradients

    def generate_signal_gradient_one_time(self, time: Array) -> Array:
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
        phase_fac = jnp.exp(-1j * self.__phase.get_value())
        lo_out = jnp.squeeze(self.__lo.compute_output(time), axis=0)
        sig = self.__complex_signal(time)
        gradients = jnp.zeros(shape=(0,))

        # Collect gradients for envelopes
        for dev in self.__envs:
            grad = jnp.squeeze(dev.compute_gradient(time).conj(), axis=0)
            if grad.size != 0:
                grad *= lo_out * phase_fac
            gradients = jnp.append(gradients, 0.5 * jnp.real(grad), axis=0)

        # Collect LO gradients
        lo_freq = self.__lo.get_parameters()[0]
        if self._is_optimized(lo_freq):
            gradients = jnp.append(gradients, 0.5j * time * sig, axis=0)

        # Collect gradient of Phase
        if self._is_optimized(self.__phase):
            gradients = jnp.append(
                gradients,
                -0.5j * sig,
                axis=0,
            )
        return gradients
