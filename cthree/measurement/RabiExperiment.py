import numpy as np

from cthree.Quantity import Quantity
from cthree.measurement.Measurement import Measurement


class RabiExperiment(Measurement):
    __qubit_freq: Quantity
    __amp: Quantity
    __freq: Quantity
    __time: Quantity

    def __init__(self, qubit_freq: float) -> None:
        self.__qubit_freq = Quantity(qubit_freq)
        self.__amp = Quantity(0.6)
        self.__freq = Quantity(0.6)
        self.__time = Quantity(0.6)

    def getParameters(self):
        return [
            self.__amp,
            self.__freq,
            self.__time
        ]

    def measure(self):
        q_freq = self.__qubit_freq.getValue()
        amp = self.__amp.getValue() * 100e6 * 2 * np.pi
        freq = self.__freq.getValue() * q_freq
        t = self.__time.getValue() * 10e-9
        diff_sq = (q_freq - freq) ** 2
        return 1 - np.abs(np.cos(np.sqrt(diff_sq + amp**2) / 2 * t) / np.sqrt(
            1 + diff_sq / (amp**2)
        )) ** 2
