from typing import List
from cthree.Quantity import Quantity
from cthree.signal.Device import Device

import jax.numpy as np
from jax import grad, vmap 


class CosToneAD(Device):
    """
    Create a simple cosine tone.
    Return the gradients calculated using Automatic Differentiation (AD).
    Right now the computeGradients expects an array of input and gives
    error if only one value of time is specified.
    """

    __amplitude: Quantity
    __frequency: Quantity

    def __init__(self) -> None:
        self.__amplitude = Quantity(
            2e5 * 2 * np.pi,
            min_value=1e5 * 2 * np.pi,
            max_value=250e6 * 2 * np.pi,
            unit="Hz",
            name="Amplitude"
        )
        self.__frequency = Quantity(
            5e9 * 2 * np.pi,
            min_value=4e9 * 2 * np.pi,
            max_value=6e9 * 2 * np.pi,
            unit="Hz",
            name="Frequency"
        )

    def getParameters(self) -> List[Quantity]:
        return [self.__amplitude, self.__frequency]

    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        return self.__computeOutputJAX(amp, freq, t)

    def __computeOutputJAX(self, amp, freq, t):
        return amp * np.cos(freq * t)

    def computeGradient(self, t: np.ndarray) -> np.ndarray:
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        grads = grad(self.__computeOutputJAX, argnums=(0, 1))
        partial_grad = lambda x: grads(amp, freq, x)
        return np.array(vmap(partial_grad)(t)) * np.array([
                                                    [self.__amplitude.getScale()],
                                                    [self.__frequency.getScale()]
                                                ])