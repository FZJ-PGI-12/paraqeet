from abc import abstractmethod
import numpy as np

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
            60e6 * 2 * np.pi, min_value=1e6 * 2 * np.pi, max_value=150e6 * 2 * np.pi
        )
        self.__frequency = Quantity(
            5e9 * 2 * np.pi, min_value=4e9 * 2 * np.pi, max_value=6e9 * 2 * np.pi
        )

    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        return amp * np.cos(freq * t)


class ZeroTone(Device):
    """
    Create a zero tone.
    """
    
    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        return np.zeros_like(t)
