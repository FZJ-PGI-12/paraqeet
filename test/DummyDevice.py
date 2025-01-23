"""Testing the device functions."""

from functools import partial

import numpy as np
import jax.numpy as jnp

from jax import Array, jit
from jax.scipy.special import erf

from cthree.signal.Envelopes import Envelope


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
        amp : cthree.Quantity
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
        amp : cthree.Quantity
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
    def _evaluateTFinalGrad(self, amp: np.ndarray, t_final: np.ndarray, t: np.ndarray):
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : cthree.Quantity
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
