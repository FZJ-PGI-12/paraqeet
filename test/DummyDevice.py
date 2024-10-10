"""Testing the device functions."""

from functools import partial

import numpy as np
import jax.numpy as jnp

from jax import jit
from jax.scipy.special import erf

from cthree.Quantity import Quantity
from cthree.signal.Waveform import Waveform


class CosToneAD(Waveform):
    """Dummy CosTone class without analytical gradients to test AD gradients."""

    __amplitude: Quantity
    __frequency: Quantity

    def __init__(self) -> None:
        self.__amplitude = Quantity(
            2e5 * 2 * np.pi,
            min_value=1e5 * 2 * np.pi,
            max_value=50e6 * 2 * np.pi,
            unit="Hz",
            name="Amplitude",
        )
        self.__frequency = Quantity(
            5e9 * 2 * np.pi,
            min_value=4e9 * 2 * np.pi,
            max_value=6e9 * 2 * np.pi,
            unit="Hz",
            name="Frequency",
        )
        self.__phase = Quantity(
            0,
            min_value=-np.pi,
            max_value=np.pi,
            unit="rad",
            name="Phase",
        )

    def getParameters(self) -> list[Quantity]:
        """Get parameters."""
        return [self.__amplitude, self.__frequency, self.__phase]

    @partial(jit, static_argnums=(0,))
    def _evaluate(
        self, amp: Quantity, freq: Quantity, phase: Quantity, t: np.ndarray
    ):
        """Compute the output of a device.

        The device explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : cthree.Quantity
            Cosine pulse amplitude.
        freq : cthree.Quantity
            Cosine pulse frequency.
        phase : cthree.Quantity
            Cosine pulse frequence.
        t : numpy.ndarray
            One-dimensional vector containing timestamps.

        """
        return jnp.squeeze(amp * jnp.cos(freq * t + phase))

    def computeOutput(self, t: np.ndarray):
        """Return the scalar output for each step in the time array t.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector containing timestamps.

        Returns
        -------
        np.ndarray
            Array of shape [t] with 't' as time.

        """
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        phase = self.__phase.getValue()
        return self._evaluate(amp, freq, phase, t)


class CosToneErfAD(Waveform):
    """Dummy CosToneErf class without analytical gradients.

    For testing AD gradients.
    """

    __amplitude: Quantity
    __frequency: Quantity
    __t_final: Quantity

    def __init__(self) -> None:
        self.__amplitude = Quantity(
            2e5 * 2 * np.pi,
            min_value=1e5 * 2 * np.pi,
            max_value=250e6 * 2 * np.pi,
            unit="Hz",
            name="Amplitude",
        )
        self.__frequency = Quantity(
            5e9 * 2 * np.pi,
            min_value=4e9 * 2 * np.pi,
            max_value=6e9 * 2 * np.pi,
            unit="Hz",
            name="Frequency",
        )
        self.__t_final = Quantity(
            10e-9, min_value=0e-9, max_value=100e-9, unit="s", name="Gate time"
        )

    def getParameters(self) -> list[Quantity]:
        """Get paramters."""
        return [self.__amplitude, self.__frequency]

    def _envelope(self, t):
        """Return a normalized error function shaped envelope with ramps.

        Ramps centered at 1/5 and 4/5 of the final gate time.

        """
        t0 = self.__t_final.getValue()
        ramp_time = t0 / 10
        rampUp = 1 + erf((t - t0 / 5) / ramp_time)
        rampDown = 1 + erf((-t + 4 * t0 / 5) / ramp_time)
        return rampUp * rampDown / 4

    @partial(jit, static_argnums=(0,))
    def _evaluate(self, amp: Quantity, freq: Quantity, t: np.ndarray):
        """Compute the output of a device.

        The device explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : cthree.Quantity
            Cosine pulse amplitude.
        freq : cthree.Quantity
            Cosine pulse frequency.
        t : numpy.ndarray
            One-dimensional time array.

        """
        return jnp.squeeze(self._envelope(t) * amp * jnp.cos(freq * t))

    def computeOutput(self, t: np.ndarray):
        """Compute the outpute."""
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        return self._evaluate(amp, freq, t)
