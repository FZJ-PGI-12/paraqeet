"""Class definition for the Device model."""

from abc import abstractmethod
from collections.abc import Callable
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
    """Classical electronics."""

    _gradientFunction: Callable | None = None
    _gradArgNums: tuple[int, ...] = ()

    @abstractmethod
    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        """Compute the output.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Output of the computation.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    @abstractmethod
    def _evaluate(self):
        """Evaluate the output of the system.

        Abstract method.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    def _computeGradientFunction(
        self,
        signalFunction: Callable,
        argnums: tuple[int, ...],
        vmap_axes: tuple[int, ...],
    ) -> Callable:
        """Return a compute gradient function from the signal function.

        Parameters
        ----------
        signalFunction : Callable
            A function that generated signals.
        argnums : Tuple[int, ...]
            A tuple of ints containing a variable number of argument numbers.
        vmap_axes : Tuple[int, ...]
            A tuple of ints.

        Returns
        -------
        Callable
            Returns a function that can be used to compute the gradient.

        """
        grads = grad(signalFunction, argnums=argnums)
        partial_grads = vmap(grads, vmap_axes)
        return jit(partial_grads)

    def setOptimisableParameters(self, params: list[Quantity]) -> None:
        """Set optimisable parameters for optimisation.

        Parameters
        ----------
        params : List[cthree.Quantity]
            Input list of parameters to be set.

        """
        super().setOptimisableParameters(params)

        Optimisable_params = self.getParameters()
        for i, param in enumerate(Optimisable_params):
            if self._isOptimised(param):
                self._gradArgNums += (i,)

    def computeGradient(self, t: np.ndarray) -> np.ndarray:
        """Compute the gradient of the `_evaluate` method.

        Uses Automatic differentiation.
        The `_evaluate` method should be a `pure` function (should take the
        optimisable parameters as function arguments and doesn't depend on
        global variables).
        Refer to https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
        for functionally `pure` functions.
        To implement analytical gradients / other methods for gradient
        computation overwrite this method in the inherited class.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Returns the gradient array of the `_evaluate` method.

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
    """Create a simple cosine tone."""

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
        """Get the amplitude of the system.

        Returns
        -------
        cthree.Quantity
            Amplitude of the system.

        """
        return self.__amplitude

    @amplitude.setter
    def amplitude(self, amplitude: Quantity) -> None:
        """Set the amplitude of the system.

        Parameters
        ----------
        cthree.Quantity
            Amplitude value of the system to be set.

        """
        self.__amplitude = amplitude

    @property
    def frequency(self) -> Quantity:
        """Get the frequency of the system.

        Returns
        -------
        cthree.Quantity
            Frequency of the system.

        """
        return self.__frequency

    @frequency.setter
    def frequency(self, frequency: Quantity) -> None:
        """Set the frequency of the system.

        Parameters
        ----------
        cthree.Quantity
            Frequency value of the system to be set.

        """
        self.__frequency = frequency

    def getParameters(self) -> list[Quantity]:
        """Get a list of parameters of the system.

        Returns
        -------
        List[Quantity]
            List of parameters of the system.

        """
        return [self.__amplitude, self.__frequency, self.__phase]

    @partial(jit, static_argnums=(0,))
    def _evaluate(
        self, amp: Quantity, freq: Quantity, phase: Quantity, t: np.ndarray
    ) -> np.ndarray:
        """Compute the output of device.

        Output explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : cthree.Quantity
            Cosine pulse amplitude.
        freq : cthree.Quantity
            Cosine pulse frequency.
        phase : cthree.Quantity
            Cosine pulse phase.
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Returns the output from the device.

        """
        return jnp.squeeze(amp * jnp.cos(freq * t + phase))

    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        """Return the scalar output for each step in the time array t.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Array of shape [t] with 't' as time.

        """
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        phase = self.__phase.getValue()
        return self._evaluate(amp, freq, phase, t)

    def computeGradient(self, t: np.ndarray) -> np.ndarray:
        """Compute the gradient of the system.

        Returns the gradient wrt dimensionless parameters for each step
        in the time array `t`.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Array of shape [t, p] with 't' as time and 'p' as the number of
            parameters.

        """
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        phase = self.__phase.getValue()
        t = jnp.array(t, ndmin=1)

        grads = []
        if self._isOptimised(self.__amplitude):
            grads.append(
                jnp.cos(freq * t + phase) * self.__amplitude.getScale()
            )
        if self._isOptimised(self.__frequency):
            grads.append(
                -amp
                * t
                * jnp.sin(freq * t + phase)
                * self.__frequency.getScale()
            )
        if self._isOptimised(self.__phase):
            grads.append(
                -amp * jnp.sin(freq * t + phase) * self.__phase.getScale()
            )
        return (
            jnp.stack(grads, axis=1)
            if len(grads) > 0
            else jnp.empty((t.shape[0], 0))
        )


class CosToneErf(Device):
    """Create a cosine tone with a fixed error-function shaped envelope."""

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
        """Get the amplitude of the system.

        Returns
        -------
        cthree.Quantity
            Amplitude of the system.

        """
        return self.__amplitude

    @amplitude.setter
    def amplitude(self, amplitude: Quantity) -> None:
        """Set the amplitude of the system.

        Parameters
        ----------
        cthree.Quantity
            Amplitude value of the system to be set.

        """
        self.__amplitude = amplitude

    @property
    def frequency(self) -> Quantity:
        """Get the frequency of the system.

        Returns
        -------
        cthree.Quantity
            Frequency of the system.

        """
        return self.__frequency

    @frequency.setter
    def frequency(self, frequency: Quantity) -> None:
        """Set the frequency of the system.

        Parameters
        ----------
        cthree.Quantity
            Frequency value of the system to be set.

        """
        self.__frequency = frequency

    @property
    def phase(self) -> Quantity:
        """Get the phase of the system.

        Returns
        -------
        cthree.Quantity
            Phase of the system.

        """
        return self.__phase

    @phase.setter
    def phase(self, phase: Quantity) -> None:
        """Set the phase of the system.

        Parameters
        ----------
        cthree.Quantity
            Phase value of the system to be set.

        """
        self.__phase = phase

    @property
    def t_final(self) -> Quantity:
        """Get the final time value of the system.

        Returns
        -------
        cthree.Quantity
            Final time value of the system.

        """
        return self.__t_final

    @t_final.setter
    def t_final(self, t_final: Quantity) -> None:
        """Set the final time value of the system.

        Parameters
        ----------
        cthree.Quantity
            Final time value of the system to be set.

        """
        self.__t_final = t_final

    def getParameters(self) -> list[Quantity]:
        """Get a list of parameters of the system.

        Returns
        -------
        List[Quantity]
            List of parameters of the system.

        """
        return [self.__amplitude, self.__frequency, self.__phase]

    def _envelope(self, t: np.ndarray) -> jnp.ndarray:
        """Create a normalised error function shaped envelope.

        Normalized, error function shaped envelope with ramps centered
        at 1/5 and 4/5 of the final gate time.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.numpy.ndarray
            Returns a envelop vector.

        """
        t0 = self.__t_final.getValue()
        ramp_time = t0 / 10
        rampUp = 1 + erf((t - t0 / 5) / ramp_time)
        rampDown = 1 + erf((-t + 4 * t0 / 5) / ramp_time)
        return rampUp * rampDown / 4

    @partial(jit, static_argnums=(0,))
    def _evaluate(
        self, amp: Quantity, freq: Quantity, phase: Quantity, t: np.ndarray
    ):
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : cthree.Quantity
            Cosine pulse amplitude.
        freq : cthree.Quantity
            Cosine pulse frequency.
        phase : cthree.Quantity
            Phase of the pulse between (-pi, pi).
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.numpy.ndarray
            Returns the output of the device that explicitly depends
            on the optimisable parameters.

        """
        return jnp.squeeze(self._envelope(t) * amp * jnp.cos(freq * t + phase))

    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        """Get the output of the device on time stamps.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Returns the output of the device.

        """
        amp = self.__amplitude.getValue()
        freq = self.__frequency.getValue()
        phase = self.__phase.getValue()
        return self._evaluate(amp, freq, phase, t)

    def computeGradient(self, t: np.ndarray) -> np.ndarray:
        """Return the gradient wrt dimensionless parameters.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
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
            jnp.stack(grads, axis=1)
            if len(grads) > 0
            else jnp.empty((t.shape[0], 0))
        )


class ZeroTone(Device):
    """Create a zero tone."""

    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        """Create a zero tone signal from an input time vector.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Returns a zero vector signal.

        """
        return jnp.zeros_like(t)
