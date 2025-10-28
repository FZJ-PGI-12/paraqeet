"""Class definition of a Rabi experiment model."""

import jax.numpy as jnp

from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Array, Quantity


class RabiExperiment(NormalizableMeasurement, Optimizable):
    """Analytic model of the general Rabi formula.

    Parameters
    ----------
    qubit_freq : Quantity
        Resonance of the single qubit.

    """

    __qubit_freq: Quantity
    __amp: Quantity
    __freq: Quantity

    def __init__(self, qubit_freq: float) -> None:
        self.__qubit_freq = Quantity(qubit_freq, 0.0, 10e9)
        self.__amp = Quantity(60e6, 0, 100e6, "Hz")
        self.__freq = Quantity(0.6 * qubit_freq, 0, 10e9)

    def get_parameters(self) -> list[Quantity]:
        """Return a list of parameters accessible in this measurement.

        Returns
        -------
        List[Quantity]
            List of parameters accessible in this measurement.

        """
        return [self.__amp, self.__freq]

    def measure(self, times: Array) -> Array | float:
        """Return measurement in the range [0, 1]."""
        return self.calculate_normalized_scalar(times)

    # TODO: calculate_normalized_scalar() will return an Array for an Array of times.
    # TODO: Make it time (float) instead? -- This is not general
    # TODO: Or should we make it Array | float?
    # TODO: For now I added a float conversion which would raise error for multiple time points.
    def calculate_normalized_scalar(self, times: Array) -> float:
        """Carry out a measurement operation.

        Gives the result of a general Rabi oscillation,
        depending of drive frequency, amplitude and time.

        Returns
        -------
        Array
            Result of a general Rabi oscillation.

        """
        q_freq = self.__qubit_freq.get_value()
        amp = self.__amp.get_value() * 2 * jnp.pi
        freq = self.__freq.get_value()
        diff_sq = (q_freq - freq) ** 2
        return float(jnp.abs(jnp.cos(jnp.sqrt(diff_sq + amp**2) / 2 * times) / jnp.sqrt(1 + diff_sq / (amp**2))) ** 2)
