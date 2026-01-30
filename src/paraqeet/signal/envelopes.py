"""Class definition for the Evelopes."""

from abc import abstractmethod
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.special import erf

from paraqeet.quantity import Array, Quantity
from paraqeet.signal.waveform import Waveform

jax.config.update("jax_enable_x64", True)


class Envelope(Waveform):
    """Classical Signal Envelope class.

    _amplitude: Quantity
        The amplitude of the envelope.
    _t_final: Quantity
        The length in time of the envelope.
    _gradient_function: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _grad_arg_nums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    _amplitude: Quantity
    _t_final: Quantity

    def __init__(
        self,
        amplitude: Quantity | None = None,
        t_final: Quantity | None = None,
    ):
        self._amplitude = amplitude or Quantity(
            1.55e8,
            min_value=jnp.array(0.0),
            max_value=jnp.array(1e9),
            unit="Hz",
            name="Amplitude",
            two_pi=True,
        )

        self._t_final = t_final or Quantity(
            32e-9,
            min_value=jnp.array(0),
            max_value=jnp.array(100e-9),
            unit="s",
            name="t_final",
        )

        self._gradient_function: Callable | None = None
        self._grad_arg_nums: tuple[int, ...] = ()

    def get_parameters(self):
        """Get a list of parameters of the envelope.

        Returns
        -------
        List[Quantity]
            List of parameters of the envelope.

        """
        return [self._amplitude, self._t_final]

    @property
    def amplitude(self) -> Quantity:
        """Get the amplitude of the system.

        Returns
        -------
        Quantity
            Amplitude of the system.

        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, amplitude: Quantity) -> None:
        """Set the amplitude of the system.

        Parameters
        ----------
        Quantity
            Amplitude value of the system to be set.

        """
        self._amplitude = amplitude

    @property
    def t_final(self) -> Quantity:
        """Get the length of the tone.

        Returns
        -------
        Quantity
            Length in time of the tone.

        """
        return self._t_final

    @t_final.setter
    def t_final(self, t_final: Quantity) -> None:
        """Set the length of the tone.

        Parameters
        ----------
        Quantity
            Length in time of the tone to be set.

        """
        self._t_final = t_final

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
    def get_value(self, times: Array | float) -> Array:
        """Compute the output.

        Parameters
        ----------
        times: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Output of the computation.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()


class ConstantEnvelope(Envelope):
    """A constant envelope tone with a fixed length.

    _amplitude: Quantity
        The amplitude of the envelope.
    _t_final: Quantity
        The length in time of the envelope.
    _gradientFunction: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _grad_arg_nums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    @partial(jit, static_argnums=(0,))
    def _evaluate(
        self,
        amp: Array,
        t_final: Array,
        t: Array | float,
    ) -> Array:
        """Evaluate the envelope depending on all parameters.

        Abstract method.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        return jnp.squeeze(jnp.where(t <= t_final, amp, 0.0))

    def get_value(self, times: Array | float) -> Array:
        """Compute the constant signal envelope at different times.

        Parameters
        ----------
        times: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Output of the computation.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        amp = self.amplitude.get_value()
        t_final = self.t_final.get_value()
        return self._evaluate(amp, t_final, times)  # type: ignore

    def get_time_gradient(self, times: Array | float) -> Array:
        """Compute a signal envelopes time derivative.

        Parameters
        ----------
        times: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns a vector signals time derivative.
        """
        return jnp.zeros_like(times)


class ZeroEnvelope(ConstantEnvelope):
    """Shorthand implentation of a zero signal envelope.

    _amplitude: Quantity
        The amplitude of the envelope.
    _t_final: Quantity
        The length in time of the envelope.
    _gradientFunction: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _grad_arg_nums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    def __init__(self):
        super().__init__()
        self.amplitude.set_value(0.0)


class FlatTopGaussianEnvelope(Envelope):
    """A flat-top Gaussian envelope.

    _amplitude: Quantity
        The amplitude of the envelope.
    _t_final: Quantity
        The length in time of the envelope.
    _gradient_function: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _grad_arg_nums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    @partial(jit, static_argnums=(0,))
    def _evaluate(self, amp: Array, t_final: Array, t: Array | float):  # type: ignore
        """Compute the output of the device.

        Explicitly depends on the optimizable parameters.

        Parameters
        ----------
        amp: Quantity
            Cosine pulse amplitude.
        t_final: Array
            The length in time of the entire envelope.
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the output of the device that explicitly depends
            on the optimizable parameters.

        """
        ramp_time = t_final / 10
        ramp_up = 1 + erf((t - t_final / 5) / ramp_time)
        ramp_down = 1 + erf((-t + 4 * t_final / 5) / ramp_time)
        return amp * ramp_up * ramp_down / 4

    @staticmethod
    @jit
    def __dir_erf(x: Array):
        return 2 / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2))

    @partial(jit, static_argnums=(0,))
    def _evaluate_time_grad(self, amp: Array, t_final: Array, t: Array):
        """Compute the output of the device.

        Explicitly depends on the optimizable parameters.

        Parameters
        ----------
        amp: Quantity
            Cosine pulse amplitude.
        t_final: Array
            The length in time of the entire envelope.
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the output of the device that explicitly depends
            on the optimizable parameters.

        """
        ramp_time = t_final / 10

        ramp_up = 1 + erf((t - t_final / 5) / ramp_time)
        ramp_up_t_dir = self.__dir_erf((t - t_final / 5) / ramp_time)
        ramp_up_t_dir /= ramp_time

        ramp_down = 1 + erf((-t + 4 * t_final / 5) / ramp_time)
        ramp_down_t_dir = self.__dir_erf((-t + 4 * t_final / 5) / ramp_time)
        ramp_down_t_dir *= -1 / ramp_time

        prod_dir = ramp_up * ramp_down_t_dir + ramp_up_t_dir * ramp_down

        return amp * prod_dir / 4

    @partial(jit, static_argnums=(0,))
    def _evaluate_t_final_grad(self, amp: Array, t_final: Array, t: Array):
        """Compute the output of the device.

        Explicitly depends on the optimizable parameters.

        Parameters
        ----------
        amp: Quantity
            Cosine pulse amplitude.
        t_final: Array
            The length in time of the entire envelope.
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the output of the device that explicitly depends
            on the optimizable parameters.

        """
        ramp_time = t_final / 10

        ramp_up = 1 + erf((t - t_final / 5) / ramp_time)
        ramp_up_t_fin_dir = self.__dir_erf((t - t_final / 5) / ramp_time)
        ramp_up_t_fin_dir *= -1 / (5 * ramp_time)

        ramp_down = 1 + erf((-t + 4 * t_final / 5) / ramp_time)
        ramp_down_t_fin_dir = self.__dir_erf((-t + 4 * t_final / 5) / ramp_time)
        ramp_down_t_fin_dir *= 4 / (5 * ramp_time)

        prod_dir = ramp_up * ramp_down_t_fin_dir + ramp_up_t_fin_dir * ramp_down

        return amp * prod_dir / 4

    def get_value(self, times: Array | float) -> Array:
        """Get the output of the device on time stamps.

        Parameters
        ----------
        times: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the output of the device.

        """
        amp = self.amplitude.get_value()
        t_final = self.t_final.get_value()
        # returns JitWrapped
        return self._evaluate(amp, t_final, times)  # type: ignore

    def get_value_and_gradient(self, times: Array | float) -> tuple[Array, Array]:
        """Return the gradient wrt dimensionless parameters.

        Parameters
        ----------
        times: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the gradient wrt dimensionless parameters.

        """
        amp = self.amplitude.get_value()
        t_final = self.t_final.get_value()
        t_arr = jnp.array(times, ndmin=1)

        grads = []
        if self._is_optimized(self.amplitude):
            grads.append(self._evaluate(jnp.array([1.0]), t_final, t_arr))
        if self._is_optimized(self.t_final):
            grads.append(self._evaluate_t_final_grad(amp, t_final, t_arr))
        gradient = jnp.stack(grads, axis=1) if len(grads) > 0 else jnp.empty((t_arr.shape[0], 0))
        return self._evaluate(amp, t_final, times), gradient

    def get_time_gradient(self, times: Array | float) -> Array:
        """Compute a signal envelopes time derivative.

        Parameters
        ----------
        times: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns a vector signals time derivative.
        """
        amp = self.amplitude.get_value()
        t_final = self.t_final.get_value()
        return jnp.array(self._evaluate_time_grad(amp, t_final, times))


class GaussEnvelope(Envelope):
    """Create a simple Gauss envelope.

    _amplitude: Quantity
        The amplitude of the envelope.
    _t_final: Quantity
        The length in time of the envelope.
    _gradient_function: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _grad_arg_nums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate(self, amp: Array, t_final: Array, t: Array) -> Array:  # type: ignore
        """Calculate the unscaled gaussian signal.

        Parameters
        ----------
        t_final: Array
            Duration of the signal to calculate the center of the gaussian from.
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            The unscaled gaussian signal.
        """
        sigma = t_final / 8
        env = amp * jnp.exp(-(1 / 2) * (t - t_final / 2) ** 2 / sigma**2)
        return jnp.squeeze(env)

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate_time_gradient(self, amp: Array, t_final: Array, t: Array) -> Array:
        """Calculate the unscaled gaussian signal.

        Parameters
        ----------
        t_final: Array
            Duration of the signal to calculate the center of the gaussian from.
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            The unscaled gaussian signals time derivative.
        """
        sigma = t_final / 8
        time_grad = self._evaluate(amp, t_final, t) * -1.0 * (t - t_final / 2) / sigma**2
        # returns JitWrapped
        return time_grad  # type: ignore

    def get_value(self, times: Array | float) -> Array:
        """Compute a Gaussian signal.

        Parameters
        ----------
        times: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns a vector gaussian signal.
        """
        t_final = self.t_final.get_value()
        amp = self.amplitude.get_value()
        # returns JitWrapped
        return self._evaluate(amp, t_final, times)  # type: ignore

    def get_time_gradient(self, times: Array | float) -> Array:
        """Compute a Gaussian signals time derivative.

        Parameters
        ----------
        times: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns a vector gaussian signals time derivative.
        """
        t_final = self.t_final.get_value()
        amp = self.amplitude.get_value()
        env_time_deriv = self._evaluate_time_gradient(amp, t_final, times)
        # returns JitWrapped
        return env_time_deriv  # type: ignore
