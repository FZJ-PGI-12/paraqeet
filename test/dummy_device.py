"""Testing the device functions."""

from functools import partial

import numpy as np
import jax.numpy as jnp

from paraQeet.quantity import Array
from jax import jit
from jax.scipy.special import erf

from paraQeet.signal.envelopes import Envelope


class FlatTopGaussianEnvelopeAD(Envelope):
    """A flat-top Gaussian envelope without analytic gradients.

    Dummy device to test AutoDiff Gradients for Envelopes.

    __amplitude: Quantity
        The amplitude of the envelope.
    __t_final: Quantity
        The length in time of the envelope.
    _gradientFunction: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _grad_arg_nums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    @partial(jit, static_argnums=(0,))
    def _evaluate(self, amp: Array, t_final: Array, t: Array):
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : paraQeet.quantity
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
        return 2 / jnp.sqrt(np.pi) * jnp.exp(-(x**2))

    @partial(jit, static_argnums=(0,))
    def _evaluateTimeGrad(self, amp: Array, t_final: Array, t: Array):
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : paraQeet.quantity
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
    def _evaluateTFinalGrad(self, amp: Array, t_final: Array, t: Array):
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : paraQeet.quantity
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
        return jnp.array(self._evaluate(amp, t_final, t))

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
