"""Testing the device functions."""

from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.scipy.special import erf

from paraqeet.quantity import Array, Quantity
from paraqeet.signal.envelopes import Envelope


class FlatTopGaussianEnvelopeAD(Envelope):
    """A flat-top Gaussian envelope without analytic gradients.

    Dummy device to test AutoDiff Gradients for Envelopes.
    """

    def __init__(
        self,
        amplitude: Quantity | None = None,
        t_up: Quantity | None = None,
        t_down: Quantity | None = None,
        ramp_time: Quantity | None = None,
        t_final: Quantity | None = None,
    ):
        """
        Args:
            amplitude: The amplitude of the envelope.
            t_up: The start time of constant section of the envelope.
            t_down: The end time of constant section of the envelope.
            ramp_time: The rate of ramp up and ramp down of the envelope.
            t_final: The length in time of the envelope. Used only if any of `t_up`, `t_down`,
                and `ramp_time` are not provided.
        """
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

        self._t_up = t_up or Quantity(
            self._t_final.get_value() / 5,
            min_value=self._t_final.get_min_value(),
            max_value=self._t_final.get_max_value(),
            unit="s",
            name="t_up",
        )

        self._t_down = t_down or Quantity(
            4 * self._t_final.get_value() / 5,
            min_value=self._t_final.get_min_value(),
            max_value=self._t_final.get_max_value(),
            unit="s",
            name="t_down",
        )

        self._ramp_time = ramp_time or Quantity(
            self._t_final.get_value() / 10,
            min_value=self._t_final.get_min_value(),
            max_value=self._t_final.get_max_value(),
            unit="s",
            name="ramp_time",
        )

    def get_parameters(self):
        return [self._amplitude, self._t_up, self._t_down, self._ramp_time]

    @partial(jit, static_argnums=(0,))
    def _evaluate(self, amp: Array, t_up: Array, t_down: Array, ramp_time: Array, times: Array):
        """Compute the output of the device.

        Explicitly depends on the optimizable parameters.

        Args:
            amp: Cosine pulse amplitude.
            t_up: The start time of constant section of the envelope.
            t_down: The end time of constant section of the envelope.
            ramp_time: The rate of ramp up and ramp down of the envelope.
            times: Array of times.

        Returns:
            The output of the device that explicitly depends on the optimizable parameters.
        """
        ramp_up = 1 + erf((times - t_up) / ramp_time)
        ramp_down = 1 + erf((-times + t_down) / ramp_time)
        return amp * ramp_up * ramp_down / 4

    def get_value(self, times: Array | float) -> Array:
        amp = self._amplitude.get_value()
        t_up = self._t_up.get_value()
        t_down = self._t_down.get_value()
        ramp_time = self._ramp_time.get_value()
        # returns JitWrapped
        return self._evaluate(amp, t_up, t_down, ramp_time, times)  # type: ignore
