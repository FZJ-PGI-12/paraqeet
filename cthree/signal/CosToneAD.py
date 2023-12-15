from abc import abstractmethod
from typing import List, Callable
from cthree.Optimisable import Optimisable
from cthree.Quantity import Quantity

import jax.numpy as np
from jax import grad, vmap


class DeviceAD(Optimisable):
    """
    Classical electronics.
    """

    @abstractmethod
    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def gradientFunction(
        self, signalFunction: Callable, argnums: tuple[int, ...], vmap_axes: tuple
    ) -> Callable:
        grads = grad(signalFunction, argnums=argnums)
        partial_grads = vmap(grads, vmap_axes)
        return partial_grads


class CosToneAD(DeviceAD):
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
        return self.__computeOutput(amp, freq, t)

    def __computeOutput(self, amp, freq, t):
        return amp * np.cos(freq * t)

    def computeGradient(self, t: np.ndarray) -> np.ndarray:
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        partial_grad = self.gradientFunction(
            self.__computeOutput, argnums=(0, 1), vmap_axes=(None, None, 0)
        )
        return np.array(partial_grad(amp, freq, t)) * np.array(
            [[self.__amplitude.getScale()], [self.__frequency.getScale()]]
        )
