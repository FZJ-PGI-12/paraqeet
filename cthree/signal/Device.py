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
            2e5 * 2 * np.pi, min_value=1e5 * 2 * np.pi, max_value=250e6 * 2 * np.pi
        )
        self.__frequency = Quantity(
            5e9 * 2 * np.pi, min_value=4e9 * 2 * np.pi, max_value=6e9 * 2 * np.pi
        )

    def getParameters(self) -> List[Quantity]:
        return [self.__amplitude, self.__frequency]

    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        return amp * np.cos(freq * t)

    def computeGradient(self, t: np.ndarray) -> List[np.ndarray]:
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        return [
            np.cos(freq * t) * self.__amplitude.getScale(),
            -amp * t * np.sin(freq * t) * self.__frequency.getScale(),
        ]


class CosToneErf(Device):
    """
    Create a simple cosine tone.
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
        )
        self.__frequency = Quantity(
            5e9 * 2 * np.pi,
            min_value=4e9 * 2 * np.pi,
            max_value=6e9 * 2 * np.pi,
            unit="Hz",
        )
        self.__t_final = Quantity(10e-9, min_value=0e-9, max_value=100e-9, unit="s")
        self.__slope = 1e9

    def getParameters(self) -> List[Quantity]:
        return [self.__amplitude, self.__frequency, self.__t_final]

    def __envelope(self, t):
        t0 = self.__t_final.getValue()
        rampUp = 1 + erf((t - t0 / 5) * self.__slope)
        rampDown = 1 + erf((-t + 4 * t0 / 5) * self.__slope)
        return rampUp * rampDown / 4

    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        return self.__envelope(t) * amp * np.cos(freq * t)

    def computeGradient(self, t: np.ndarray) -> List[np.ndarray]:
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
