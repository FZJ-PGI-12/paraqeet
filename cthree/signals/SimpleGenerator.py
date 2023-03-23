from typing import List
import numpy as np

from cthree.Quantity import Quantity
from cthree.signals.Generator import Generator


class CosGenerator(Generator):
    """
    Simple sinusodial signal generation.
    """

    __devices: List

    def __init__(self, devices: List):
        self.__devices = devices
        self.__amplitude = Quantity(0.6)
        self.__frequency = Quantity(0.6)

    def generateSignal(self, t):
        """
        Generate a signal for time(s) t.
        """
        amp = self.__amplitude.get_value() * 100e6 * 2 * np.pi
        freq = self.__frequency.get_value() * 5e9 * 2 * np.pi
        return amp * np.cos(freq * t)
