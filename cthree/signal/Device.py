from abc import abstractmethod
from typing import List
import numpy as np

from scipy.special import erf

from cthree.Quantity import Quantity
from cthree.Optimisable import Optimisable


class Device(Optimisable):
    """
    Classical electronics.
    """

    @abstractmethod
    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class CosTone(Device):
    """
    Create a simple cosine tone.
    """

    __amplitude: Quantity
    __frequency: Quantity

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

    def getParameters(self) -> List[Quantity]:
        return [self.__amplitude, self.__frequency]

    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        return amp * np.cos(freq * t)

    def computeGradient(self, t: np.ndarray) -> List[np.ndarray]:
        """
        Returns the gradient wrt dimensionless parameters.
        """
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        return [
            np.cos(freq * t) * self.__amplitude.getScale(),
            -amp * t * np.sin(freq * t) * self.__frequency.getScale(),
        ]


class CosToneErf(Device):
    """
    Create a simple cosine tone with a fixed, error-function shaped envelope.
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
        return [self.__amplitude, self.__frequency, self.__t_final]

    def __envelope(self, t):
        """
        Normalized, error function shaped envelope with ramps centered at 1/5 and 4/5 of the final gate time.
        """
        t0 = self.__t_final.getValue()
        ramp_time = t0 / 10
        rampUp = 1 + erf((t - t0 / 5) / ramp_time)
        rampDown = 1 + erf((-t + 4 * t0 / 5) / ramp_time)
        return rampUp * rampDown / 4

    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        return self.__envelope(t) * amp * np.cos(freq * t)

    def computeGradient(self, t: np.ndarray) -> List[np.ndarray]:
        """
        Returns the gradient wrt dimensionless parameters.
        """
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        dc_dAmp = np.cos(freq * t) * self.__envelope(t)
        dc_dFreq = -amp * t * np.sin(freq * t) * self.__envelope(t)
        return [
            self.__amplitude.getScale() * dc_dAmp,
            self.__frequency.getScale() * dc_dFreq,
        ]


class ZeroTone(Device):
    """
    Create a zero tone.
    """

    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        return np.zeros_like(t)
