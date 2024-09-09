from abc import abstractmethod
from typing import List, Callable, Tuple
import numpy as np

from cthree.Quantity import Quantity
from cthree.Optimisable import Optimisable

from jax.scipy.special import erf

import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from functools import partial

jax.config.update("jax_enable_x64", True)


class Device(Optimisable):
    """
    Classical electronics.
    """

    _gradientFunction: Callable | None = None
    _gradArgNums: Tuple[int, ...] = ()

    @abstractmethod
    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _evaluate(self):
        raise NotImplementedError()

    def _computeGradientFunction(
        self,
        signalFunction: Callable,
        argnums: Tuple[int, ...],
        vmap_axes: Tuple[int, ...],
    ) -> Callable:
        grads = grad(signalFunction, argnums=argnums)
        partial_grads = vmap(grads, vmap_axes)
        return jit(partial_grads)

    def setOptimisableParameters(self, params: List[Quantity]) -> None:
        super().setOptimisableParameters(params)

        Optimisable_params = self.getParameters()
        for i, param in enumerate(Optimisable_params):
            if self._isOptimised(param):
                self._gradArgNums += (i,)

    def computeGradient(self, t: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the `_evaluate` method using Automatic differentiation.
        The `_evaluate` method should be a `pure` function (should take the
        optimisable parameters as function arguments and doesn't depend on global variables).

        Refer to https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
        for functionally `pure` functions.

        To implement analytical gradients / other methods for gradient computation
        overwrite this method in the inherited class.
        """

        params = self.getParameters()
        param_values = [param.getValue() for param in params]
        num_params = len(params)
        t = jnp.array(t, ndmin=1)

        # vmap over time axis only, set everything else to None
        vmap_axes = (None,) * num_params
        vmap_axes += (0,)  # type: ignore

        grads = jnp.empty((t.shape[0], 0))

        if len(self._gradArgNums) > 0:
            if self._gradientFunction is None:
                self._gradientFunction = self._computeGradientFunction(
                    self._evaluate,
                    argnums=self._gradArgNums,
                    vmap_axes=vmap_axes,
                )

            parameter_scales = jnp.array([param.getScale() for param in params])
            parameter_scales = jnp.reshape(
                parameter_scales, (1,) + parameter_scales.shape
            )

            grads = jnp.stack(self._gradientFunction(*param_values, t), axis=1)
            grads = jnp.squeeze(grads) * parameter_scales

        return grads


class CosTone(Device):
    """
    Create a simple cosine tone.
    """

    __amplitude: Quantity
    __frequency: Quantity
    __phase: Quantity

    def __init__(self) -> None:
        self.__amplitude = Quantity(
            2e5 * 2 * np.pi,
            min_value=1e5 * 2 * np.pi,
            max_value=50e6 * 2 * np.pi,
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
        self.__phase = Quantity(
            0,
            min_value=-np.pi,
            max_value=np.pi,
            unit="rad",
            name="Phase",
        )

    @property
    def amplitude(self) -> Quantity:
        return self.__amplitude

    @amplitude.setter
    def amplitude(self, amplitude: Quantity) -> None:
        self.__amplitude = amplitude

    @property
    def frequency(self) -> Quantity:
        return self.__frequency

    @frequency.setter
    def frequency(self, frequency: Quantity) -> None:
        self.__frequency = frequency

    def getParameters(self) -> List[Quantity]:
        return [self.__amplitude, self.__frequency, self.__phase]

    @partial(jit, static_argnums=(0,))
    def _evaluate(
        self, amp: Quantity, freq: Quantity, phase: Quantity, t: np.ndarray
    ) -> np.ndarray:
        """
        Function to compute the output of the device that explicitly depends on the optimisable parameters.

        Args:
            amp (Quantity): Cosine pulse amplitude
            freq (Quantity): Cosine pulse frequency
            t (np.ndarray): Time array
        """
        return jnp.squeeze(amp * jnp.cos(freq * t + phase))

    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        """
        Returns the scalar output for each step in the time array t.

        Returns:
            np.ndarray: array of shape [t] with t: time
        """
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        phase = self.__phase.getValue()
        return self._evaluate(amp, freq, phase, t)

    def computeGradient(self, t: np.ndarray) -> np.ndarray:
        """
        Returns the gradient wrt dimensionless parameters for each step in the time array t.

        Returns:
            np.ndarray: array of shape [t, p] with t: time, p: number of parameter
        """
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        phase = self.__phase.getValue()
        t = jnp.array(t, ndmin=1)

        grads = []
        if self._isOptimised(self.__amplitude):
            grads.append(jnp.cos(freq * t + phase) * self.__amplitude.getScale())
        if self._isOptimised(self.__frequency):
            grads.append(
                -amp * t * jnp.sin(freq * t + phase) * self.__frequency.getScale()
            )
        if self._isOptimised(self.__phase):
            grads.append(-amp * jnp.sin(freq * t + phase) * self.__phase.getScale())
        return (
            jnp.stack(grads, axis=1) if len(grads) > 0 else jnp.empty((t.shape[0], 0))
        )


class CosToneErf(Device):
    """
    Create a simple cosine tone with a fixed, error-function shaped envelope.
    """

    __amplitude: Quantity
    __frequency: Quantity
    __phase: Quantity
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
        self.__phase = Quantity(
            0,
            min_value=-np.pi,
            max_value=np.pi,
            unit="rad",
            name="Phase",
        )
        self.__t_final = Quantity(
            10e-9, min_value=0e-9, max_value=100e-9, unit="s", name="Gate time"
        )

    @property
    def amplitude(self) -> Quantity:
        return self.__amplitude

    @amplitude.setter
    def amplitude(self, amplitude: Quantity) -> None:
        self.__amplitude = amplitude

    @property
    def frequency(self) -> Quantity:
        return self.__frequency

    @frequency.setter
    def frequency(self, frequency: Quantity) -> None:
        self.__frequency = frequency

    @property
    def phase(self) -> Quantity:
        return self.__phase

    @phase.setter
    def phase(self, phase: Quantity) -> None:
        self.__phase = phase

    @property
    def t_final(self) -> Quantity:
        return self.__t_final

    @t_final.setter
    def t_final(self, t_final: Quantity) -> None:
        self.__t_final = t_final

    def getParameters(self) -> List[Quantity]:
        return [self.__amplitude, self.__frequency, self.__phase]

    def _envelope(self, t):
        """
        Normalized, error function shaped envelope with ramps centered at 1/5 and 4/5 of the final gate time.
        """
        t0 = self.__t_final.getValue()
        ramp_time = t0 / 10
        rampUp = 1 + erf((t - t0 / 5) / ramp_time)
        rampDown = 1 + erf((-t + 4 * t0 / 5) / ramp_time)
        return rampUp * rampDown / 4

    @partial(jit, static_argnums=(0,))
    def _evaluate(self, amp: Quantity, freq: Quantity, phase: Quantity, t: np.ndarray):
        """
        Function to compute the output of the device that explicitly depends on the optimisable parameters.

        Args:
            amp (Quantity): Cosine pulse amplitude
            freq (Quantity): Cosine pulse frequency
            phase (Quantity): Phase of the pulse between (-pi, pi)
            t (np.ndarray): Time array
        """
        return jnp.squeeze(self._envelope(t) * amp * jnp.cos(freq * t + phase))

    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        phase = self.__phase.getValue()
        return self._evaluate(amp, freq, phase, t)

    def computeGradient(self, t: np.ndarray) -> np.ndarray:
        """
        Returns the gradient wrt dimensionless parameters.
        """
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        phase = self.__phase.getValue()
        t = jnp.array(t, ndmin=1)

        grads = []
        if self._isOptimised(self.__amplitude):
            dc_dAmp = jnp.cos(freq * t) * self._envelope(t)
            grads.append(self.__amplitude.getScale() * dc_dAmp)
        if self._isOptimised(self.__frequency):
            dc_dFreq = -amp * t * jnp.sin(freq * t) * self._envelope(t)
            grads.append(self.__frequency.getScale() * dc_dFreq)
        if self._isOptimised(self.__phase):
            grads.append(
                -amp
                * self._envelope(t)
                * jnp.sin(freq * t + phase)
                * self.__phase.getScale()
            )

        return (
            jnp.stack(grads, axis=1) if len(grads) > 0 else jnp.empty((t.shape[0], 0))
        )


class ZeroTone(Device):
    """
    Create a zero tone.
    """

    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        return jnp.zeros_like(t)
