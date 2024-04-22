from typing import List
import numpy as np
import jax.numpy as jnp

from numpy import ndarray
from cthree.measurement.Measurement import Measurement
from cthree.Exceptions import ConfigurationException


class WeightedSumGoal(Measurement):
    """
    Combine multiple measurements into a single goal function.
    """

    __measurements: List[Measurement]
    __weights: np.ndarray

    def __init__(self, measurements: List[Measurement], weights: np.ndarray):
        self.__measurements = measurements
        self.__weights = weights
        if len(measurements) != len(weights):
            raise ConfigurationException(
                f"Incompatible number of measurements {len(measurements)} and weights {len(weights)}"
            )
        if not np.isclose(sum(weights), 1.0):
            raise UserWarning("Supplied weights are not normalized.")

    def measure(self) -> ndarray:
        measurements = [m.measure() for m in self.__measurements]
        sumMeas = 0
        for ii, w in enumerate(self.__weights):
            sumMeas += w * measurements[ii]
        return sumMeas

    def measureNormalised(self) -> ndarray:
        measurements = [m.measureNormalised() for m in self.__measurements]
        sumMeas = 0
        for ii, w in enumerate(self.__weights):
            sumMeas += w * measurements[ii]
        return float(sumMeas)

    def measureWithGradient(self):
        measurements = [m.measureWithGradient() for m in self.__measurements]
        sumMeas = jnp.array(0)
        sumGrads = jnp.zeros_like(measurements[0][1])
        for ii, w in enumerate(self.__weights):
            sumMeas += w * measurements[ii][0]
            sumGrads += w * measurements[ii][1]
        return sumMeas, sumGrads
