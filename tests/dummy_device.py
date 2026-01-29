"""Testing the device functions."""

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.special import erf

from paraqeet.quantity import Array
from paraqeet.signal.envelopes import Envelope


class FlatTopGaussianEnvelopeAD(Envelope):
    """A flat-top Gaussian envelope without analytic gradients.

    Dummy device to test AutoDiff Gradients for Envelopes.

    __amplitude: Quantity
        The amplitude of the envelope.
    __t_final: Quantity
        The length in time of the envelope.
    _gradient_function: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _grad_arg_nums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    @partial(jit, static_argnums=(0,))
    def _evaluate(self, amp: Array, t_final: Array, t: Array):
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
        return 2 / jnp.sqrt(np.pi) * jnp.exp(-(x**2))

    @partial(jit, static_argnums=(0,))
    def _evaluate_time_grad(self, amp: Array, t_final: Array, t: Array):
        """Compute the output of the device.

        Explicitly depends on the optimizable parameters.

        Parameters
        ----------
        amp  Quantity
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
    def _evaluate_tfinal_grad(self, amp: Array, t_final: Array, t: Array):
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

    def get_value(self, t: Array) -> Array:
        """Get the output of the device on time stamps.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the output of the device.

        """
        amp = self.amplitude.get_value()
        t_final = self.t_final.get_value()
        return jnp.array(self._evaluate(amp, t_final, t))

    def get_time_gradient(self, t: Array) -> Array:
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
        return jnp.array(self._evaluate_time_grad(amp, t_final, t))
