from abc import abstractmethod
from typing import List, Callable
from cthree.Optimisable import Optimisable
from cthree.Quantity import Quantity

import jax
import jax.numpy as np
from jax import grad, vmap
from jax.scipy.special import erf
from functools import partial

jax.config.update("jax_enable_x64", True)


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

    @partial(jax.jit, static_argnums=(0,))
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


class CosToneErfAD(DeviceAD):
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

    @partial(jax.jit, static_argnums=(0,))
    def __computeOutput(self, amp, freq, t):
        return self.__envelope(t) * amp * np.cos(freq * t)

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
        return self.__computeOutput(amp, freq, t)

    def computeGradient(self, t: np.ndarray) -> List[np.ndarray]:
        """
        Returns the gradient wrt dimensionless parameters.
        """
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        partial_grad = self.gradientFunction(
            self.__computeOutput, argnums=(0, 1), vmap_axes=(None, None, 0)
        )

        if np.shape(t) == ():
            t = np.array([t])

        return np.array(partial_grad(amp, freq, t)) * np.array(
            [
                [self.__amplitude.getScale()],
                [self.__frequency.getScale()],
            ]
        )
