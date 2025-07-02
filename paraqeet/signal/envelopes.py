"""Class definition for the Evelopes."""

from abc import abstractmethod
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from paraqeet.quantity import Array
from jax import jit
from jax.scipy.special import erf

from paraqeet.quantity import Quantity
from paraqeet.signal.waveform import Waveform

jax.config.update("jax_enable_x64", True)


class Envelope(Waveform):
    """Classical Signal Envelope class.

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

        self._gradientFunction: Callable | None = None
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
        paraqeet.quantity
            Amplitude of the system.

        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, amplitude: Quantity) -> None:
        """Set the amplitude of the system.

        Parameters
        ----------
        paraqeet.Quantity
            Amplitude value of the system to be set.

        """
        self._amplitude = amplitude

    @property
    def t_final(self) -> Quantity:
        """Get the length of the tone.

        Returns
        -------
        paraqeet.quantity
            Length in time of the tone.

        """
        return self._t_final

    @t_final.setter
    def t_final(self, t_final: Quantity) -> None:
        """Set the length of the tone.

        Parameters
        ----------
        paraqeet.Quantity
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
    def compute_output(self, t: Array) -> Array:
        """Compute the output.

        Parameters
        ----------
        t : Array
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

    def compute_output(self, t: Array) -> Array:
        """Compute the constant signal envelope at different times.

        Parameters
        ----------
        t : Array
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
        return self._evaluate(amp, t_final, t)  # type: ignore

    def compute_time_gradient(self, t: Array) -> Array:
        """Compute a signal envelopes time derivative.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns a vector signals time derivative.
        """
        return jnp.zeros_like(t)


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
    _gradientFunction: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _grad_arg_nums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    @partial(jit, static_argnums=(0,))
    def _evaluate(self, amp: Array, t_final: Array, t: Array | float):  # type: ignore
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : Quantity
            Cosine pulse amplitude.
        t_final: Array
            The length in time of the entire envelope.
        t : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        chtree.quantity.Array
            Returns the output of the device that explicitly depends
            on the optimisable parameters.

        """
        ramp_time = t_final / 10
        rampUp = 1 + erf((t - t_final / 5) / ramp_time)
        rampDown = 1 + erf((-t + 4 * t_final / 5) / ramp_time)
        return amp * rampUp * rampDown / 4

    @staticmethod
    @jit
    def __dir_erf(x: Array):
        return 2 / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2))

    @partial(jit, static_argnums=(0,))
    def _evaluateTimeGrad(self, amp: Array, t_final: Array, t: Array):
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : Quantity
            Cosine pulse amplitude.
        t_final: Array
            The length in time of the entire envelope.
        t : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        chtree.quantity.Array
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
    def _evaluate_t_final_grad(self, amp: Array, t_final: Array, t: Array):
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : Quantity
            Cosine pulse amplitude.
        t_final: Array
            The length in time of the entire envelope.
        t : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        chtree.quantity.Array
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

    def compute_output(self, t: Array) -> Array:
        """Get the output of the device on time stamps.

        Parameters
        ----------
        t : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the output of the device.

        """
        amp = self.amplitude.get_value()
        t_final = self.t_final.get_value()
        return self._evaluate(amp, t_final, t)  # type: ignore

    def compute_gradient(self, t: Array) -> Array:
        """Return the gradient wrt dimensionless parameters.

        Parameters
        ----------
        t : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the gradient wrt dimensionless parameters.

        """
        amp = self.amplitude.get_value()
        t_final = self.t_final.get_value()
        t_arr = jnp.array(t, ndmin=1)

        grads = []
        if self._is_optimised(self.amplitude):
            grads.append(self._evaluate(jnp.array([1.0]), t_final, t_arr))
        if self._is_optimised(self.t_final):
            grads.append(self._evaluate_t_final_grad(amp, t_final, t_arr))
        return jnp.stack(grads, axis=1) if len(grads) > 0 else jnp.empty((t_arr.shape[0], 0))

    def compute_time_gradient(self, t: Array) -> Array:
        """Compute a signal envelopes time derivative.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns a vector signals time derivative.
        """
        amp = self.amplitude.get_value()
        t_final = self.t_final.get_value()
        return jnp.array(self._evaluateTimeGrad(amp, t_final, t))


class GaussEnvelope(Envelope):
    """Create a simple Gauss envelope.

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

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate(self, amp: Array, t_final: Array, t: Array) -> Array:  # type: ignore
        """Calculate the unscaled gaussian signal.

        Parameters
        ----------
        t_final : Array
            Duration of the signal to calculate the center of the gaussian from.
        t : Array
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
        t_final : Array
            Duration of the signal to calculate the center of the gaussian from.
        t : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            The unscaled gaussian signals time derivative.
        """
        sigma = t_final / 8
        timeGrad = self._evaluate(amp, t_final, t) * -1.0 * (t - t_final / 2) / sigma**2
        return timeGrad  # type: ignore

    def compute_output(self, t: Array) -> Array:
        """Compute a Gaussian signal.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns a vector gaussian signal.
        """
        t_final = self.t_final.get_value()
        amp = self.amplitude.get_value()
        return self._evaluate(amp, t_final, t)  # type: ignore

    def compute_time_gradient(self, t: Array) -> Array:
        """Compute a Gaussian signals time derivative.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns a vector gaussian signals time derivative.
        """
        t_final = self.t_final.get_value()
        amp = self.amplitude.get_value()
        envTimeDeriv = self._evaluate_time_gradient(amp, t_final, t)
        return envTimeDeriv  # type: ignore
