"""Class definition for the Sinusoidal generator model."""

import numpy as np
import jax.numpy as jnp
from jax import Array

from cthree.Quantity import Quantity
from cthree.signal.Waveform import Waveform, LocalOscillator
from cthree.signal.Generator import Generator


class CosGenerator(Generator):
    """Simple sinusoidal signal generation.

    Parameters
    ----------
    envelopes : List[cthree.signal.Waveform]
        List of input devices.

    """

    __envs: list[Waveform]
    __phase: Quantity
    _optimisableParameters: list[Quantity] = []

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

    def getParameters(self) -> list[Quantity]:
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
            pars += env.getParameters()
        pars += self.__lo.getParameters()
        pars += [self.__phase]
        return pars

    def setOptimisableParameters(self, params: list[Quantity]) -> None:
        """Set specified parameters to be optimised.

        Also add the indices to `__gradArgNums` to compute the gradients.

        Parameters
        ----------
        params : list[Quantity]
        """
        super().setOptimisableParameters(params)

        for dev in self.__envs:
            dev.setOptimisableParameters(params)

        self.__lo.setOptimisableParameters(params)

    def generateSignal(self, t: np.ndarray) -> Array:
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
        env = jnp.zeros_like(t)
        for dev in self.__envs:
            env += jnp.reshape(dev.computeOutput(t), env.shape)
        sig = env.conj() * self.__lo.computeOutput(t)
        sig = sig * jnp.exp(-1j * self.__phase.getValue())
        return jnp.real(sig)

    def generateSignalGradient(self, t) -> Array:
        """Collect and returns the gradients from all devices.

        TODO - Check the following formulas for derivatives

        Here, since the signal = Re(env(t).conj() * LO(t) * exp(-i*phase))
        Derivative of the signal wrt optimisable parameter of envelope would be
        Re(denv(t).conj() * LO(t) * e^(-i*phase))

        And derivative of signal wrt parameter of LO would be
        Re(env(t).conj() * dLO(t) * e^(-i*phase))

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.Array
            Returns the signal gradient vector.

        """
        phase_fac = jnp.exp(-1j * self.__phase.getValue())
        lo_out = self.__lo.computeOutput(t)
        gradients = jnp.zeros(shape=(t.shape[0], 0))

        # Collect gradients for envelopes
        for dev in self.__envs:
            grad = dev.computeGradient(t).conj() * lo_out * phase_fac
            gradients = jnp.append(gradients, jnp.real(grad), axis=1)

        # Collect gradient of LO
        total_env = jnp.zeros_like(t)
        for dev in self.__envs:
            total_env += jnp.reshape(dev.computeOutput(t), total_env.shape)
        gradLO = total_env.conj() * self.__lo.computeGradient(t) * phase_fac
        gradients = jnp.append(gradients, jnp.real(gradLO), axis=1)

        # Collect gradient of Phase
        if self._isOptimised(self.__phase):
            sig = self.generateSignal(t)
            gradients = jnp.append(
                gradients,
                jnp.expand_dims(
                    -1.0j * sig * phase_fac * self.__phase.getScale(), 1
                ),
                axis=1,
            )
        return gradients

    def generateSignalGradientOneTime(self, t) -> Array:
        """Return the gradients from all devices at the given time.

        Parameters
        ----------
        t : float
            Single timestamp.

        Returns
        -------
        jax.Array
            Return the gradients from all devices at one time.

        """
        # TODO: I think we were and still are missing the product of the
        #  gradient of the LO signal with the envelope signal
        phase_fac = jnp.exp(-1j * self.__phase.getValue())
        lo_out = self.__lo.computeOutput(t)
        gradients = jnp.zeros(shape=(0,))

        # Collect gradients for envelopes
        for dev in self.__envs:
            grad = jnp.squeeze(
                dev.computeGradient(t).conj() * lo_out * phase_fac, axis=0
            )
            gradients = jnp.append(gradients, jnp.real(grad), axis=0)

        # Collect gradient of LO
        total_env = jnp.zeros_like(t)
        for dev in self.__envs:
            total_env += jnp.reshape(dev.computeOutput(t), total_env.shape)
        gradLO = jnp.squeeze(
            total_env.conj() * self.__lo.computeGradient(t) * phase_fac, axis=0
        )
        gradients = jnp.append(gradients, jnp.real(gradLO), axis=0)

        # Collect gradient of Phase
        if self._isOptimised(self.__phase):
            sig = self.generateSignal(t)
            gradients = jnp.append(
                gradients,
                -1.0j * sig * phase_fac * self.__phase.getScale(),
                axis=0,
            )
        return gradients
