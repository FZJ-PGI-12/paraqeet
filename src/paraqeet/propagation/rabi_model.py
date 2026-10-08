"""Class definition of a Rabi experiment model."""

import jax.numpy as jnp

from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Array, Quantity


class RabiModel(Optimizable):
    """Analytic model of the general Rabi formula."""

    _qubit_freq: Quantity
    _amp: Quantity
    _freq: Quantity

    def __init__(self, qubit_freq: float) -> None:
        """
        Args:
            qubit_freq: Resonance of the single qubit.
        """
        self._qubit_freq = Quantity(qubit_freq, 0.0, 10e9)
        self._amp = Quantity(60e6, 0, 100e6, "Hz")
        self._freq = Quantity(0.6 * qubit_freq, 0, 10e9)

    def get_parameters(self) -> list[Quantity]:
        """Return a list of parameters accessible in this measurement.

        Returns:
            List of parameters accessible in this measurement.
        """
        return [self._amp, self._freq]

    def get_value(self, times: Array) -> Array:
        """Returns the analytic value state.

        Gives the result of a general Rabi oscillation,
        depending on drive frequency, amplitude and time.

        Note:
            Returns the measurement value at the last time point.

        Returns:
            Result of a general Rabi oscillation.
        """
        q_freq = self._qubit_freq.get_value()
        amp = self._amp.get_value() * 2 * jnp.pi
        freq = self._freq.get_value()
        diff_sq = (q_freq - freq) ** 2
        norm = jnp.sqrt(1 + diff_sq / (amp**2))
        phase = jnp.sqrt(diff_sq + amp**2) / 2 * times
        return jnp.array([jnp.sin(phase), jnp.cos(phase)]) / norm
