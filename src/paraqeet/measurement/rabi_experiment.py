"""Class definition of a Rabi experiment model."""

import jax.numpy as jnp

from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Array, Float, Quantity


class RabiExperiment(NormalizableMeasurement, Optimizable):
    """Analytic model of the general Rabi formula.

    Parameters
    ----------
    qubit_freq : Quantity
        Resonance of the single qubit.

    """

    _qubit_freq: Quantity
    _amp: Quantity
    _freq: Quantity

    def __init__(self, qubit_freq: float) -> None:
        self._qubit_freq = Quantity(qubit_freq, 0.0, 10e9)
        self._amp = Quantity(60e6, 0, 100e6, "Hz")
        self._freq = Quantity(0.6 * qubit_freq, 0, 10e9)

    def get_parameters(self) -> list[Quantity]:
        """Return a list of parameters accessible in this measurement.

        Returns
        -------
        List[Quantity]
            List of parameters accessible in this measurement.

        """
        return [self._amp, self._freq]

    def measure(self, times: Array) -> Array | Float:
        """Return measurement in the range [0, 1]."""
        return self.calculate_normalized_scalar(times)

    def calculate_normalized_scalar(self, times: Array) -> Float:
        """Carry out a measurement operation.

        Gives the result of a general Rabi oscillation,
        depending of drive frequency, amplitude and time.

        *Note: Returns the measumement value at the last time point.*

        Returns
        -------
        Array
            Result of a general Rabi oscillation.

        """
        t = times if isinstance(times, float) else times[-1]
        q_freq = self._qubit_freq.get_value()
        amp = self._amp.get_value() * 2 * jnp.pi
        freq = self._freq.get_value()
        diff_sq = (q_freq - freq) ** 2
        norm = jnp.sqrt(1 + diff_sq / (amp**2))
        return float(jnp.abs(jnp.cos(jnp.sqrt(diff_sq + amp**2) / 2 * t) / norm**2).item())
