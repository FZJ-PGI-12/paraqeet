from typing import List
import jax.numpy as np

from numpy import ndarray
from cthree.measurement.Measurement import Measurement


class MultiGoal:
    """
    Combine multiple measurements into a single goal function.
    """

    __measurements: List[Measurement]
    __weights: np.ndarray

    def __init__(self, meas: List[Measurement], weights: np.ndarray):
        self.__measurements = meas
        self.__weights = weights

    def measure(self) -> ndarray:
        measurements = [m.measure() for m in self.__measurements]
        sumMeas = 0
        for ii, w in enumerate(self.__weights):
            sumMeas += w * measurements[ii]
        return sumMeas

    def measureWithGradient(self):
        measurements = [m.measureWithGradient() for m in self.__measurements]
        sumMeas = np.array(0)
        sumGrads = np.zeros_like(measurements[0][1])
        for ii, w in enumerate(self.__weights):
            sumMeas += w * measurements[ii][0]
            sumGrads += w * measurements[ii][1]
        return sumMeas, sumGrads
