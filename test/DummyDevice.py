from functools import partial
from typing import List

import numpy as np
import jax.numpy as jnp

from jax import jit
from jax.scipy.special import erf

from cthree.Quantity import Quantity
from cthree.signal.Device import Device


class CosToneAD(Device):
    """
    Dummy CosTone class without analytical gradients to test AD gradients
    """

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

    def getParameters(self) -> List[Quantity]:
        return [self.__amplitude, self.__frequency, self.__phase]

    @partial(jit, static_argnums=(0,))
    def _evaluate(self, amp: Quantity, freq: Quantity, phase: Quantity, t: np.ndarray):
        """
        Function to compute the output of the device that explicitly depends on the optimisable parameters.

        Args:
            amp (Quantity): Cosine pulse amplitude
            freq (Quantity): Cosine pulse frequency
            t (np.ndarray): Time array
        """
        return jnp.squeeze(amp * jnp.cos(freq * t + phase))

    def computeOutput(self, t: np.ndarray):
        """
        Returns the scalar output for each step in the time array t.

        Returns:
            np.ndarray: array of shape [t] with t: time
        """
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        phase = self.__phase.getValue()
        return self._evaluate(amp, freq, phase, t)


class CosToneErfAD(Device):
    """
    Dummy CosToneErf class without analytical gradients to test AD gradients
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

    def getParameters(self) -> List[Quantity]:
        return [self.__amplitude, self.__frequency]

    def _envelope(self, t):
        """
        Normalized, error function shaped envelope with ramps centered at 1/5 and 4/5 of the final gate time.
        """
        t0 = self.__t_final.getValue()
        ramp_time = t0 / 10
        rampUp = 1 + erf((t - t0 / 5) / ramp_time)
        rampDown = 1 + erf((-t + 4 * t0 / 5) / ramp_time)
        return rampUp * rampDown / 4

    @partial(jit, static_argnums=(0,))
    def _evaluate(self, amp: Quantity, freq: Quantity, t: np.ndarray):
        """
        Function to compute the output of the device that explicitly depends on the optimisable parameters.

        Args:
            amp (Quantity): Cosine pulse amplitude
            freq (Quantity): Cosine pulse frequency
            t (np.ndarray): Time array
        """
        return jnp.squeeze(self._envelope(t) * amp * jnp.cos(freq * t))

    def computeOutput(self, t: np.ndarray):
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        return self._evaluate(amp, freq, t)
