"""Class definition for the Sinusoidal generator model."""

import numpy as np
import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.scipy.special import erf


from cthree.quantity import Quantity
from cthree.signal.waveform import Waveform, LocalOscillator
from cthree.signal.generator import Generator
from cthree.signal.envelopes import Envelope


class IQMixer(Generator):
    """Control signal generation.

    Waveforms of envelopes (low bandwith) are mixed with a local oscillator
    (high bandwidth) to apply a desired control field to the system.

    Parameters
    ----------
    envelopes : List[cthree.signal.Waveform]
        List of input devices.

    """

    __envs: list[Waveform]
    __phase: Quantity
    _optimisable_parameters: list[Quantity] = []

    def __init__(
        self,
        envelopes: list[Waveform] | None,
        frequency: Quantity | None = None,
        phase: Quantity | None = None,
    ):
        self.__envs = envelopes or []

        self.__lo = LocalOscillator(frequency=frequency)

        self.__phase = phase or Quantity(
            np.array(0.0),
            min_value=np.array(-np.pi),
            max_value=np.array(np.pi),
            unit="rad",
            name="Phase",
        )

    def get_parameters(self) -> list[Quantity]:
        """Return a list of parameters.

        Collects and returns a list of parameters from the tone, generator
        and the carrier signal.

        Returns
        -------
        List[Quantity]
            All Parameters describing the signal.
        """
        pars = []
        for env in self.__envs:
            pars += env.get_parameters()
        pars += self.__lo.get_parameters()
        pars += [self.__phase]
        return pars

    def set_optimisable_parameters(self, params: list[Quantity]) -> None:
        """Set specified parameters to be optimised.

        Parameters
        ----------
        params : list[Quantity]
        """
        super().set_optimisable_parameters(params)

        for dev in self.__envs:
            dev.set_optimisable_parameters(params)

        self.__lo.set_optimisable_parameters(params)

    def __complex_signal(self, t: np.ndarray) -> Array:
        """Generate a signal for time(s) 't'.

        Doesnt take real value now for ease of gradient computation.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.Array
            Returns the signal vector.

        """
        env = jnp.zeros_like(t)
        for dev in self.__envs:
            env += jnp.reshape(dev.compute_output(t), env.shape)
        sig = env.conj() * self.__lo.compute_output(t)
        sig = sig * jnp.exp(-1j * self.__phase.get_value())
        return sig

    def generate_signal(self, t: np.ndarray) -> Array:
        """Generate a signal for time(s) 't'.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.Array
            Returns the signal vector.

        """
        return jnp.real(self.__complex_signal(t))

    def generate_signal_gradient(self, t) -> Array:
        """Collect and returns the gradients from all devices.

        Since the signal = Re(env(t).conj() * e^(i*freq*t) * exp(-i*phase))
        Derivative of the signal wrt optimisable parameter of envelope would be
        0.5 * Re(denv(t).conj() * e^(i*freq*t) * e^(-i*phase))
        (TODO - Check the envelope derivatives)

        And derivative of signal wrt parameter of LO would be
        0.5 * Re(env(t).conj() * i*t*e^(i*freq*t) * e^(-i*phase))

        And derivative of signal wrt phase would be
        -0.5 * i * Re(env(t).conj() * e^(i*freq*t) * e^(-i*phase))

        The 0.5 are due to the Wirtinger derivatives due to Re part.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.Array
            Returns the signal gradient vector.

        """
        phase_fac = jnp.exp(-1j * self.__phase.get_value())
        lo_out = self.__lo.compute_output(t)
        sig = self.__complex_signal(t)
        gradients = jnp.zeros(shape=(t.shape[0], 0))

        # Collect gradients for envelopes
        for dev in self.__envs:
            grad = dev.compute_gradient(t).conj()
            if grad.size != 0:
                grad *= jnp.expand_dims(lo_out * phase_fac, axis=1)
            gradients = jnp.append(gradients, 0.5 * jnp.real(grad), axis=1)

        # Collect LO gradients
        lo_freq = self.__lo.get_parameters()[0]
        if self._is_optimised(lo_freq):
            gradients = jnp.append(
                gradients,
                jnp.expand_dims(0.5j * t * sig, 1),
                axis=1,
            )

        # Collect gradient of Phase
        if self._is_optimised(self.__phase):
            gradients = jnp.append(
                gradients,
                jnp.expand_dims(-0.5j * sig, 1),
                axis=1,
            )
        return gradients

    def generate_signal_gradient_one_time(self, t) -> Array:
        """Return the gradients from all devices at the given time.

        Since the signal = Re(env(t).conj() * e^(i*freq*t) * exp(-i*phase))
        Derivative of the signal wrt optimisable parameter of envelope would be
        0.5 * Re(denv(t).conj() * e^(i*freq*t) * e^(-i*phase))
        (TODO - Check the envelope derivatives)

        And derivative of signal wrt parameter of LO would be
        0.5*i*t * env(t).conj() * e^(i*freq*t) * e^(-i*phase)

        And derivative of signal wrt phase would be
        -0.5*i * env(t).conj() * e^(i*freq*t) * e^(-i*phase)

        The 0.5 are due to the Wirtinger derivatives due to Re part.

        Parameters
        ----------
        t : float
            Single timestamp.

        Returns
        -------
        jax.Array
            Return the gradients from all devices at one time.

        """
        phase_fac = jnp.exp(-1j * self.__phase.get_value())
        lo_out = jnp.squeeze(self.__lo.compute_output(t), axis=0)
        sig = self.__complex_signal(t)
        gradients = jnp.zeros(shape=(0,))

        # Collect gradients for envelopes
        for dev in self.__envs:
            grad = jnp.squeeze(dev.compute_gradient(t).conj(), axis=0)
            if grad.size != 0:
                grad *= lo_out * phase_fac
            gradients = jnp.append(gradients, 0.5 * jnp.real(grad), axis=0)

        # Collect LO gradients
        lo_freq = self.__lo.get_parameters()[0]
        if self._is_optimised(lo_freq):
            gradients = jnp.append(gradients, 0.5j * t * sig, axis=0)

        # Collect gradient of Phase
        if self._is_optimised(self.__phase):
            gradients = jnp.append(
                gradients,
                -0.5j * sig,
                axis=0,
            )
        return gradients
