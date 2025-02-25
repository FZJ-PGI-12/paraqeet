"""Class definition for the Evelopes."""

from abc import abstractmethod
from collections.abc import Callable
from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.special import erf
from jax.typing import ArrayLike as Array

from cthree.signal.waveform import Waveform
from cthree.quantity import Quantity

jax.config.update("jax_enable_x64", True)


class Envelope(Waveform):
    """Classical Signal Envelope class.

    __amplitude: Quantity
        The amplitude of the envelope.
    __t_final: Quantity
        The length in time of the envelope.
    _gradientFunction: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _gradArgNums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    def __init__(
        self,
        amplitude: Quantity | None = None,
        t_final: Quantity | None = None,
    ):
        self.__amplitude = amplitude or Quantity(
            1.55e8,
            min_value=np.array(0.0),
            max_value=np.array(1e9),
            unit="Hz",
            name="Amplitude",
            two_pi=True,
        )

        self.__t_final = t_final or Quantity(
            32e-9,
            min_value=np.array(0),
            max_value=np.array(100e-9),
            unit="s",
            name="t_final",
        )

        self._gradientFunction: Callable | None = None
        self._gradArgNums: tuple[int, ...] = ()

    def get_parameters(self):
        """Get a list of parameters of the envelope.

        Returns
        -------
        List[Quantity]
            List of parameters of the envelope.

        """
        return [self.__amplitude, self.__t_final]

    @property
    def amplitude(self) -> Quantity:
        """Get the amplitude of the system.

        Returns
        -------
        cthree.quantity
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
    def t_final(self) -> Quantity:
        """Get the length of the tone.

        Returns
        -------
        cthree.quantity
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

    @abstractmethod
    def _evaluate(self, *args, **kwargs):
        """Evaluate the output of the envelope.

        Abstract method.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    @abstractmethod
    def compute_output(self, t: np.ndarray) -> np.ndarray:
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


class ConstantEnvelope(Envelope):
    """A constant envelope tone with a fixed length.

    __amplitude: Quantity
        The amplitude of the envelope.
    __t_final: Quantity
        The length in time of the envelope.
    _gradientFunction: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _gradArgNums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    @partial(jit, static_argnums=(0,))
    def _evaluate(
        self,
        amp: np.ndarray,
        t_final: np.ndarray,
        t: np.ndarray,
    ) -> Array:
        """Evaluate the envelope depending on all parameters.

        Abstract method.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        return jnp.squeeze(jnp.where(t <= t_final, amp, 0.0))

    def compute_output(self, t: np.ndarray) -> Array:
        """Compute the constant signal envelope at different times.

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
        amp = self.amplitude.get_value()
        t_final = self.t_final.get_value()
        return self._evaluate(amp, t_final, t)

    def compute_time_gradient(self, t: np.ndarray) -> Array:
        """Compute a signal envelopes time derivative.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        np.ndarray
            Returns a vector signals time derivative.
        """
        return jnp.zeros_like(t)


class ZeroEnvelope(ConstantEnvelope):
    """Shorthand implentation of a zero signal envelope.

    __amplitude: Quantity
        The amplitude of the envelope.
    __t_final: Quantity
        The length in time of the envelope.
    _gradientFunction: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _gradArgNums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    def __init__(self):
        super().__init__()
        self.amplitude.set_value(0.0)


class FlatTopGaussianEnvelope(Envelope):
    """A flat-top Gaussian envelope.

    __amplitude: Quantity
        The amplitude of the envelope.
    __t_final: Quantity
        The length in time of the envelope.
    _gradientFunction: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _gradArgNums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    @partial(jit, static_argnums=(0,))
    def _evaluate(self, amp: np.ndarray, t_final: np.ndarray, t: np.ndarray):
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : cthree.quantity
            Cosine pulse amplitude.
        t_final: np.ndarray
            The length in time of the entire envelope.
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.numpy.ndarray
            Returns the output of the device that explicitly depends
            on the optimisable parameters.

        """
        ramp_time = t_final / 10
        rampUp = 1 + erf((t - t_final / 5) / ramp_time)
        rampDown = 1 + erf((-t + 4 * t_final / 5) / ramp_time)
        return amp * rampUp * rampDown / 4

    @staticmethod
    @jit
    def __dir_erf(x: np.ndarray):
        return 2 / jnp.sqrt(np.pi) * jnp.exp(-(x**2))

    @partial(jit, static_argnums=(0,))
    def _evaluateTimeGrad(self, amp: np.ndarray, t_final: np.ndarray, t: np.ndarray):
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : cthree.quantity
            Cosine pulse amplitude.
        t_final: np.ndarray
            The length in time of the entire envelope.
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.numpy.ndarray
            Returns the output of the device that explicitly depends
            on the optimisable parameters.

        """
        ramp_time = t_final / 10

        rampUp = 1 + erf((t - t_final / 5) / ramp_time)
        rampUp_t_dir = self.__dir_erf((t - t_final / 5) / ramp_time)
        rampUp_t_dir /= ramp_time

        rampDown = 1 + erf((-t + 4 * t_final / 5) / ramp_time)
        rampDown_t_dir = self.__dir_erf((-t + 4 * t_final / 5) / ramp_time)
        rampDown_t_dir *= -1 / ramp_time

        prod_dir = rampUp * rampDown_t_dir + rampUp_t_dir * rampDown

        return amp * prod_dir / 4

    @partial(jit, static_argnums=(0,))
    def _evaluate_t_final_grad(self, amp: np.ndarray, t_final: np.ndarray, t: np.ndarray):
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : cthree.quantity
            Cosine pulse amplitude.
        t_final: np.ndarray
            The length in time of the entire envelope.
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.numpy.ndarray
            Returns the output of the device that explicitly depends
            on the optimisable parameters.

        """
        ramp_time = t_final / 10

        rampUp = 1 + erf((t - t_final / 5) / ramp_time)
        rampUp_t_fin_dir = self.__dir_erf((t - t_final / 5) / ramp_time)
        rampUp_t_fin_dir *= -1 / (5 * ramp_time)

        rampDown = 1 + erf((-t + 4 * t_final / 5) / ramp_time)
        rampDown_t_fin_dir = self.__dir_erf((-t + 4 * t_final / 5) / ramp_time)
        rampDown_t_fin_dir *= 4 / (5 * ramp_time)

        prod_dir = rampUp * rampDown_t_fin_dir + rampUp_t_fin_dir * rampDown

        return amp * prod_dir / 4

    def compute_output(self, t: np.ndarray) -> Array:
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
        amp = self.amplitude.get_value()
        t_final = self.t_final.get_value()
        return self._evaluate(amp, t_final, t)

    def compute_gradient(self, t: np.ndarray) -> Array:
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
        amp = self.amplitude.get_value()
        t_final = self.t_final.get_value()
        t = jnp.array(t, ndmin=1)

        grads = []
        if self._is_optimised(self.amplitude):
            grads.append(self._evaluate(np.array(1.0), t_final, t))
        if self._is_optimised(self.t_final):
            grads.append(self._evaluate_t_final_grad(amp, t_final, t))
        return jnp.stack(grads, axis=1) if len(grads) > 0 else jnp.empty((t.shape[0], 0))

    def compute_time_gradient(self, t: np.ndarray) -> Array:
        """Compute a signal envelopes time derivative.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        np.ndarray
            Returns a vector signals time derivative.
        """
        amp = self.amplitude.get_value()
        t_final = self.t_final.get_value()
        return self._evaluateTimeGrad(amp, t_final, t)


class GaussEnvelope(Envelope):
    """Create a simple Gauss envelope.

    __amplitude: Quantity
        The amplitude of the envelope.
    __t_final: Quantity
        The length in time of the envelope.
    _gradientFunction: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _gradArgNums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate(self, amp: np.ndarray, t_final: np.ndarray, t: np.ndarray) -> Array:
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
        sigma = t_final / 8
        env = amp * jnp.exp(-(1 / 2) * (t - t_final / 2) ** 2 / sigma**2)
        return jnp.squeeze(env)

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate_time_gradient(self, amp: np.ndarray, t_final: np.ndarray, t: np.ndarray) -> Array:
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
        sigma = t_final / 8
        timeGrad = self._evaluate(amp, t_final, t) * -1.0 * (t - t_final / 2) / sigma**2
        return timeGrad

    def compute_output(self, t: np.ndarray) -> Array:
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
        t_final = self.t_final.get_value()
        amp = self.amplitude.get_value()
        return self._evaluate(amp, t_final, t)

    def compute_time_gradient(self, t: np.ndarray) -> Array:
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
        t_final = self.t_final.get_value()
        amp = self.amplitude.get_value()
        envTimeDeriv = self._evaluate_time_gradient(amp, t_final, t)
        return envTimeDeriv
