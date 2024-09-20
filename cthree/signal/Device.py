"""Class definition for the Device model."""

from abc import abstractmethod
from collections.abc import Callable
from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax.scipy.special import erf
from jax.typing import ArrayLike as Array

from cthree.Optimisable import Optimisable
from cthree.Quantity import Quantity

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
    ):
        """Return a compute gradient function from the signal function.

        Parameters
        ----------
        signalFunction : Callable
            A function that generated signals.
        argnums : Tuple[int, ...]
            A tuple of ints containing a variable number of argument numbers.
        vmap_axes : Tuple[int, ...]
            A tuple of ints.

        """
        grads = grad(signalFunction, argnums=argnums)
        partial_grads = vmap(grads, vmap_axes)
        self._gradientFunction = jit(partial_grads)

    def setOptimisableParameters(self, params: list[Quantity]) -> None:
        """Set optimisable parameters for optimisation.

        Parameters
        ----------
        params : List[cthree.Quantity]
            Input list of parameters to be set.

        """
        super().setOptimisableParameters(params)

        self._gradArgNums = ()
        for i, param in enumerate(self.getParameters()):
            if self._isOptimised(param):
                self._gradArgNums += (i,)

        # Recompute gradient function
        params = self.getParameters()
        num_params = len(params)

        # vmap over time axis only, set everything else to None
        vmap_axes = (None,) * num_params
        vmap_axes += (0,)  # type: ignore

        if len(self._gradArgNums) > 0:
            self._computeGradientFunction(
                self._evaluate,
                argnums=self._gradArgNums,
                vmap_axes=vmap_axes,
            )
        else:
            self._gradientFunction = None

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
        t = jnp.array(t, ndmin=1)

        grads = jnp.empty((t.shape[0], 0))

        if self._gradientFunction is not None:
            grads = jnp.stack(self._gradientFunction(*param_values, t), axis=1)

            parameter_scales = jnp.array(
                [param.getScale() for param in self._optimisableParameters]
            )

            if len(self._gradArgNums) > 1:
                parameter_scales = jnp.reshape(
                    parameter_scales, (1,) + parameter_scales.shape
                )
                grads = jnp.squeeze(grads) * parameter_scales
            else:
                grads = jnp.squeeze(grads) * parameter_scales
                grads = jnp.reshape(grads, (-1, 1))

        return grads

    @abstractmethod
    def computeEnvelope(self, t: np.ndarray) -> Array:
        """Compute a signal envelope.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        np.ndarray
            Returns a vector signal.
        """
        raise NotImplementedError(
            "This tone does not have an envelope defined!"
        )

    def computeEnvelopeTimeGradient(self, t: np.ndarray) -> Array:
        """Compute a signal envelopes time derivative.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        np.ndarray
            Returns a vector gaussian signals time derivative.
        """
        t = jnp.array(t, ndmin=1)
        envTimeGradFun = grad(self.computeEnvelope, argnums=0)
        envTimeGrad = vmap(envTimeGradFun, in_axes=(0,))(t)
        return jnp.squeeze(envTimeGrad)


class CosTone(Device):
    """
    Create a simple cosine tone.

    __amplitude:
        The Amplitude of the Cosine.
    __frequency:
        The Frequency of the Cosine.
    """

    __amplitude: Quantity
    __frequency: Quantity
    __t_final: Quantity

    def __init__(
        self,
        amplitude: Quantity | None = None,
        frequency: Quantity | None = None,
        t_final: Quantity | None = None,
    ) -> None:
        self.__amplitude = amplitude or Quantity(
            2e5 * 2 * np.pi,
            min_value=1e5 * 2 * np.pi,
            max_value=50e6 * 2 * np.pi,
            unit="Hz",
            name="Amplitude",
        )

        self.__frequency = frequency or Quantity(
            5e9 * 2 * np.pi,
            min_value=4e9 * 2 * np.pi,
            max_value=6e9 * 2 * np.pi,
            unit="Hz",
            name="Frequency",
        )

        self.__t_final = t_final or Quantity(
            value=np.array(10e-9),
            min_value=np.array(0.0),
            max_value=np.array(100e-9),
            unit="s",
            name="t_final",
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
    def t_final(self) -> Quantity:
        """Get the length of the tone.

        Returns
        -------
        cthree.Quantity
            Length in time of the tone.

        """
        return self.__t_final

    @t_final.setter
    def t_final(self, t_final: Quantity) -> None:
        """Set the length of the tone.

        Parameters
        ----------
        cthree.Quantity
            Length in time of the tone to be set.

        """
        self.__t_final = t_final

    def getParameters(self) -> list[Quantity]:
        """Get a list of parameters of the system.

        Returns
        -------
        List[Quantity]
            List of parameters of the system.

        """
        return [self.__amplitude, self.__frequency]

    @partial(jit, static_argnums=(0,))
    def _evaluate(
        self, amp: Quantity, freq: Quantity, t: np.ndarray
    ) -> np.ndarray:
        """Compute the output of device.

        Output explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : cthree.Quantity
            Cosine pulse amplitude.
        freq : cthree.Quantity
            Cosine pulse frequency.
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Returns the output from the device.

        """
        return jnp.squeeze(amp * jnp.cos(freq * t))

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
        return self._evaluate(amp, freq, t)

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
        t = jnp.array(t, ndmin=1)

        grads = []
        if self._isOptimised(self.__amplitude):
            grads.append(jnp.cos(freq * t) * self.__amplitude.getScale())
        if self._isOptimised(self.__frequency):
            grads.append(
                -amp * t * jnp.sin(freq * t) * self.__frequency.getScale()
            )
        return (
            jnp.stack(grads, axis=1)
            if len(grads) > 0
            else jnp.empty((t.shape[0], 0))
        )

    def computeEnvelope(self, t: np.ndarray) -> Array:
        """Compute a signal envelope.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        np.ndarray
            Returns a vector signal.
        """
        amp = self.__amplitude.getValue()
        return jnp.squeeze(amp * jnp.ones_like(t))


class CosToneErf(Device):
    """A simple cosine tone with a fixed, error-function shaped envelope.

    __amplitude:
        The Amplitude of the Cosine.
    __frequency:
        The Frequency of the Cosine.
    __t_final:
        The length of the entire pulse.
    """

    __amplitude: Quantity
    __frequency: Quantity
    __t_final: Quantity

    def __init__(
        self,
        amplitude: Quantity | None = None,
        frequency: Quantity | None = None,
        t_final: Quantity | None = None,
    ) -> None:
        self.__amplitude = amplitude or Quantity(
            2e5 * 2 * np.pi,
            min_value=1e5 * 2 * np.pi,
            max_value=250e6 * 2 * np.pi,
            unit="Hz",
            name="Amplitude",
        )

        self.__frequency = frequency or Quantity(
            5e9 * 2 * np.pi,
            min_value=4e9 * 2 * np.pi,
            max_value=6e9 * 2 * np.pi,
            unit="Hz",
            name="Frequency",
        )

        self.__t_final = t_final or Quantity(
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
    def t_final(self) -> Quantity:
        """Get the length of the tone.

        Returns
        -------
        cthree.Quantity
            Length in time of the tone.

        """
        return self.__t_final

    @t_final.setter
    def t_final(self, t_final: Quantity) -> None:
        """Set the length of the tone.

        Parameters
        ----------
        cthree.Quantity
            Length in time of the tone to be set.

        """
        self.__t_final = t_final

    def getParameters(self) -> list[Quantity]:
        """Get a list of parameters of the system.

        -------

        Returns
        -------
        List[Quantity]
            List of parameters of the system.

        """
        return [self.__amplitude, self.__frequency]

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
    def _evaluate(self, amp: Quantity, freq: Quantity, t: np.ndarray):
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : cthree.Quantity
            Cosine pulse amplitude.
        freq : cthree.Quantity
            Cosine pulse frequency.
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.numpy.ndarray
            Returns the output of the device that explicitly depends
            on the optimisable parameters.

        """
        return self._envelope(t) * amp * jnp.cos(freq * t)

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
        return self._evaluate(amp, freq, t)

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
        t = jnp.array(t, ndmin=1)

        grads = []
        if self._isOptimised(self.__amplitude):
            dc_dAmp = jnp.cos(freq * t) * self._envelope(t)
            grads.append(self.__amplitude.getScale() * dc_dAmp)
        if self._isOptimised(self.__frequency):
            dc_dFreq = -amp * t * jnp.sin(freq * t) * self._envelope(t)
            grads.append(self.__frequency.getScale() * dc_dFreq)
        return (
            jnp.stack(grads, axis=1)
            if len(grads) > 0
            else jnp.empty((t.shape[0], 0))
        )

    def computeEnvelope(self, t: np.ndarray) -> Array:
        """Compute a erf signal envelope.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        np.ndarray
            Returns a vector gaussian signal.
        """
        amp = self.__amplitude.getValue()
        return jnp.squeeze(amp * self._envelope(t))


class ZeroTone(Device):
    """Create a zero tone.

    __t_final: Quantity
        The length of the signal.
    """

    __t_final: Quantity

    def __init__(self, t_final: Quantity | None = None):
        self.__t_final = t_final or Quantity(
            value=np.array(10e-9),
            min_value=np.array(0.0),
            max_value=np.array(100e-9),
            unit="s",
            name="t_final",
        )

    @property
    def t_final(self) -> Quantity:
        """Get the length of the tone.

        Returns
        -------
        cthree.Quantity
            Length in time of the tone.

        """
        return self.__t_final

    @t_final.setter
    def t_final(self, t_final: Quantity) -> None:
        """Set the length of the tone.

        Parameters
        ----------
        cthree.Quantity
            Length in time of the tone to be set.

        """
        self.__t_final = t_final

    def getParameters(self) -> list[Quantity]:
        """Return device parameters.

        Returns
        -------
        list[Quantity]
        """
        return [self.__t_final]

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
        return jnp.zeros_like(t, dtype=np.complex128)

    def computeEnvelope(self, t: np.ndarray) -> Array:
        """Compute a signal envelope.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        np.ndarray
            Returns a vector signal.
        """
        return jnp.zeros_like(t, dtype=np.complex128)

    def computeEnvelopeTimeGradient(self, t: np.ndarray) -> Array:
        """Compute a signal envelopes time gradient.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        np.ndarray
            Returns a vector signal.
        """
        return jnp.zeros_like(t, dtype=np.complex128)


class CarrierTone(Device):
    """A LO carrier signal generator.

    Parameters
    ----------
    __carrier_freq : Quantity
        The frequency of the carrier signal
    """

    __carrier_freq: Quantity

    def __init__(self, carrier_freq: Quantity | None = None) -> None:
        self.__carrier_freq = carrier_freq or Quantity(
            value=np.array(4.8e9 * 2 * np.pi),
            min_value=np.array(0),
            max_value=np.array(6e9 * 2 * np.pi),
            unit="Hz",
            name="carrier_freq",
        )

    @property
    def frequency(self) -> Quantity:
        """Get The frequency of the constant oscillating tone.

        Returns
        -------
        Quantity
            The frequency of the tone.

        """
        return self.__carrier_freq

    @frequency.setter
    def frequency(self, frequency: Quantity) -> None:
        """Set The frequency of the constant oscillating tone.

        Parameters
        ----------
        freq : Quantity
            The frequency of the constant oscillating tone.

        """
        self.__carrier_freq = frequency

    def getParameters(self) -> list[Quantity]:
        """Return device parameters.

        Returns
        -------
        list[Quantity]
            Returns the carrier frequency.
        """
        return [self.__carrier_freq]

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate(self, freq: np.ndarray, t: np.ndarray) -> jnp.ndarray:
        """Calculate the unscaled carrier signal.

        Parameters
        ----------
        freq : numpy.ndarray
            The frequency of the carrier signal
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            The unscaled the carrier signal.
        """
        return jnp.exp(1j * freq * t)

    def computeOutput(self, t: np.ndarray) -> jnp.ndarray:
        """Evaluate a carrier signal from an input time vector.

        Parameters
        ----------
        freq :
            Frequency of the carrier signal
        t : np.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        np.ndarray
            Returns a vector carrier signal.
        """
        return self._evaluate(self.__carrier_freq.getValue(), t)

    def computeGradient(self, t: np.ndarray) -> np.ndarray:
        """Return the gradient wrt to frequency of carrier signal.

        Parameters
        ----------
        t : np.ndarray
            Array of time points to evaluate gradients at.

        Returns
        -------
        np.ndarray
            Gradient of tone wrt to frequency.
        """
        freq = self.__carrier_freq.getValue()
        t = jnp.array(t, ndmin=1)

        grads = jnp.empty((t.shape[0], 0))
        if self._isOptimised(self.__carrier_freq):
            dc_dFreq = 1j * t * self._evaluate(freq, t)
            grads = self.__carrier_freq.getScale() * dc_dFreq
            grads = jnp.reshape(grads, (-1, 1))

        return grads

    def computeEnvelope(self, t: np.ndarray) -> np.ndarray:
        """Not a Tone. So no Envelope function.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()


class GaussTone(Device):
    """Create a simple Gauss tone.

    Parameters
    ----------
    __amplitude:
        Amplitude of the Gaussian
    __t_final:
        Length of the signal.
    """

    __amplitude: Quantity
    __t_final: Quantity

    def __init__(
        self, amplitude: Quantity | None = None, t_final: Quantity | None = None
    ) -> None:
        self.__amplitude = amplitude or Quantity(
            np.array(3.8e08),
            min_value=np.array(0.0),
            max_value=np.array(1.0e9),
            unit="Hz",
            name="Amplitude",
        )

        self.__t_final = t_final or Quantity(
            value=np.array(10e-9),
            min_value=np.array(0.0),
            max_value=np.array(100e-9),
            unit="s",
            name="t_final",
        )

    def getParameters(self) -> list[Quantity]:
        """Return the amplitude and t_final as parameters.

        Returns
        -------
        list[Quantity]
            Amplitude and t_final of the tone.
        """
        return [self.__amplitude, self.__t_final]

    @property
    def t_final(self):
        """Length of the pulse.

        Returns
        -------
        Quantity
            Total duration of the pulse.
        """
        return self.__t_final

    @t_final.setter
    def t_final(self, t_final: Quantity) -> None:
        """Set the length of the tone.

        Parameters
        ----------
        cthree.Quantity
            Length in time of the tone to be set.

        """
        self.__t_final = t_final

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate(
        self, amp: np.ndarray, t_final: np.ndarray, t: np.ndarray
    ) -> Array:
        """Calculate the unscaled gaussian signal.

        Parameters
        ----------
        t_final : np.ndarray
            Duration of the signal to calculate the center of the gaussian from.
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            The unscaled gaussian signal.
        """
        sigma = t_final / 6
        env = amp * jnp.exp(-(1 / 2) * (t - t_final / 2) ** 2 / sigma**2)
        return jnp.squeeze(env)

    @partial(jax.jit, static_argnums=(0,))
    def _evaluateTimeGradient(
        self, amp: np.ndarray, t_final: np.ndarray, t: np.ndarray
    ) -> Array:
        """Calculate the unscaled gaussian signal.

        Parameters
        ----------
        t_final : np.ndarray
            Duration of the signal to calculate the center of the gaussian from.
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            The unscaled gaussian signals time derivative.
        """
        sigma = t_final / 6
        timeGrad = (
            self._evaluate(amp, t_final, t)
            * -1.0
            * (t - t_final / 2)
            / sigma**2
        )
        return jnp.squeeze(timeGrad)

    def computeOutput(self, t: np.ndarray) -> Array:
        """Compute a Gaussian signal.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        np.ndarray
            Returns a vector gaussian signal.
        """
        return self.computeEnvelope(t)

    def computeEnvelope(self, t: np.ndarray) -> Array:
        """Compute a Gaussian signal envelope.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        np.ndarray
            Returns a vector gaussian signal.
        """
        t_final = self.__t_final.getValue()
        amp = self.__amplitude.getValue()
        return self._evaluate(amp, t_final, t)

    def computeEnvelopeTimeGradient(self, t: np.ndarray) -> Array:
        """Compute a Gaussian signals time derivative.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        np.ndarray
            Returns a vector gaussian signals time derivative.
        """
        t_final = self.__t_final.getValue()
        amp = self.__amplitude.getValue()
        envTimeDeriv = self._evaluateTimeGradient(amp, t_final, t)
        return envTimeDeriv
