import numpy as np

from cthree.Quantity import Quantity
from cthree.Optimisable import Optimisable


class Device(Optimisable):
    """
    Classical electronics.
    """

    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        pass


class CosTone(Device):
    """
    Create a simple cosine tone.
    """

    __amplitude: Quantity
    __frequency: Quantity

    def __init__(self) -> None:
        self.__amplitude = Quantity(0.6)
        self.__frequency = Quantity(0.6)

    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        amp = self.__amplitude.get_value() * 100e6 * 2 * np.pi
        freq = self.__frequency.get_value() * 5e9 * 2 * np.pi
        return amp * np.cos(freq * t)
