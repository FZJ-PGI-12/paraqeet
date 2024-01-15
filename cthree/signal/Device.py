from abc import abstractmethod
from typing import List, Callable
import numpy as np

from cthree.Quantity import Quantity
from cthree.Optimisable import Optimisable

from scipy.special import erf

import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax.scipy.special import erf as jax_erf
from functools import partial

jax.config.update("jax_enable_x64", True)


class Device(Optimisable):
    """
    Classical electronics.
    """

    @abstractmethod
    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _computeGradientFunction(
        self, signalFunction: Callable, argnums: tuple[int, ...], vmap_axes: tuple
    ) -> Callable:
        grads = grad(signalFunction, argnums=argnums)
        partial_grads = vmap(grads, vmap_axes)
        return jit(partial_grads)


class CosTone(Device):
    """
    Create a simple cosine tone.
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

    def _evaluate(self, amp, freq, t):
        """
        Function to compute the output of the device that explicitly depends on the optimisable parameters.

        Args:
            amp (Quantity): Cosine pulse amplitude
            freq (Quantity): Cosine pulse frequency
            t (np.ndarray): Time array
        """
        return amp * np.cos(freq * t)

    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        return self._evaluate(amp, freq, t)

    def computeGradient(self, t: np.ndarray) -> List[np.ndarray]:
        """
        Returns the gradient wrt dimensionless parameters.
        """
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        return [
            np.cos(freq * t) * self.__amplitude.getScale(),
            -amp * t * np.sin(freq * t) * self.__frequency.getScale(),
        ]


class CosToneErf(Device):
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

    def _envelope(self, t):
        """
        Normalized, error function shaped envelope with ramps centered at 1/5 and 4/5 of the final gate time.
        """
        t0 = self.__t_final.getValue()
        ramp_time = t0 / 10
        rampUp = 1 + erf((t - t0 / 5) / ramp_time)
        rampDown = 1 + erf((-t + 4 * t0 / 5) / ramp_time)
        return rampUp * rampDown / 4

    def _evaluate(self, amp, freq, t):
        """
        Function to compute the output of the device that explicitly depends on the optimisable parameters.

        Args:
            amp (Quantity): Cosine pulse amplitude
            freq (Quantity): Cosine pulse frequency
            t (np.ndarray): Time array
        """
        return self._envelope(t) * amp * np.cos(freq * t)

    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        return self._evaluate(amp, freq, t)

    def computeGradient(self, t: np.ndarray) -> List[np.ndarray]:
        """
        Returns the gradient wrt dimensionless parameters.
        """
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        dc_dAmp = np.cos(freq * t) * self._envelope(t)
        dc_dFreq = -amp * t * np.sin(freq * t) * self._envelope(t)
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


class CosToneAD(CosTone):
    """
    Create a cos tone, but the gradients are calculated by Automatic Differentiation (AD).
    This class is for testing purposes and hence runs slower than analytically calculated gradients.
    """

    __gradientFunction: Callable

    def __init__(self) -> None:
        super().__init__()
        self.__gradientFunction = None

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate(self, amp, freq, t):
        """
        Overwrite the `_evaluate` function to compute the output of the device that explicitly depends on the
        optimisable parameters.
        This uses JAX based Numpy to make it compatible to AD.

        Args:
            amp (Quantity): Cosine pulse amplitude
            freq (Quantity): Cosine pulse frequency
            t (np.ndarray): Time array
        """
        return amp * jnp.cos(freq * t)

    def computeGradient(self, t: np.ndarray) -> List[jnp.ndarray]:
        """
        Overwrite the inherited `computeGradient` method to calculate gradients uisng AD.
        """
        params = self.getParameters()
        amp = params[0].getValue()
        freq = params[1].getValue()

        if self.__gradientFunction is None:
            self.__gradientFunction = self._computeGradientFunction(
                self._evaluate, argnums=(0, 1), vmap_axes=(None, None, 0)
            )
        if jnp.shape(t) == ():
            t = jnp.array([t])

        return jnp.array(self.__gradientFunction(amp, freq, t)) * jnp.array(
            [[params[0].getScale()], [params[1].getScale()]]
        )


class CosToneErfAD(CosToneErf):
    """
    Create a cos tone with error-function shaped envelope, but the gradients are calculated
    by Automatic Differentiation (AD).
    This class is for testing purposes and hence runs slower than analytically calculated gradients.
    """

    __gradientFunction: Callable | None

    def __init__(self) -> None:
        super().__init__()
        self.__gradientFunction = None

    def _envelope(self, t):
        """
        Overwrite the inherited `_envelope` function to calaculate normalized, error function shaped envelope
        with ramps centered at 1/5 and 4/5 of the final gate time.
        This uses JAX based error function to make it compatible to AD.
        """

        t_final = self.getParameters()[2]
        t0 = t_final.getValue()
        ramp_time = t0 / 10
        rampUp = 1 + jax_erf((t - t0 / 5) / ramp_time)
        rampDown = 1 + jax_erf((-t + 4 * t0 / 5) / ramp_time)
        return rampUp * rampDown / 4

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate(self, amp, freq, t):
        """
        Overwrite the inherited `_evaluate` function to compute the output of the device that explicitly
        depends on the optimisable parameters.
        This uses JAX based Numpy to make it compatible to AD.

        Args:
            amp (Quantity): Cosine pulse amplitude
            freq (Quantity): Cosine pulse frequency
            t (np.ndarray): Time array
        """
        return self._envelope(t) * amp * jnp.cos(freq * t)

    def computeGradient(self, t: np.ndarray) -> List[jnp.ndarray]:
        """
        Overwrite the inherited `computeGradient` method to calculate gradients uisng AD.
        """
        params = self.getParameters()
        amp = params[0].getValue()
        freq = params[1].getValue()

        if self.__gradientFunction is None:
            self.__gradientFunction = self._computeGradientFunction(
                self._evaluate, argnums=(0, 1), vmap_axes=(None, None, 0)
            )
        if jnp.shape(t) == ():
            t = jnp.array([t])

        return jnp.array(self.__gradientFunction(amp, freq, t)) * jnp.array(
            [[params[0].getScale()], [params[1].getScale()]]
        )
