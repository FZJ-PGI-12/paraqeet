"""Class definition of a Rabi experiment model."""

import jax.numpy as jnp

from paraqeet.measurement.measurement import Measurement
from paraqeet.quantity import Quantity


class RabiExperiment(Measurement):
    """Analytic model of the general Rabi formula.

    Parameters
    ----------
    qubit_freq : Quantity
        Resonance of the single qubit.

    """

    __qubit_freq: Quantity
    __amp: Quantity
    __freq: Quantity
    __time: Quantity

    def __init__(self, qubit_freq: float) -> None:
        super().__init__(jnp.asarray(0.0))
        self.__qubit_freq = Quantity(qubit_freq, 0.0, 10e9)
        self.__amp = Quantity(60e6, 0, 100e6, "Hz")
        self.__freq = Quantity(0.6 * qubit_freq, 0, 10e9)
        self.__time = Quantity(0.6e-9, 0, 10e-9)

    def get_parameters(self):
        """Return a list of parameters accessible in this measurement.

        Returns
        -------
        List[Quantity]
            List of parameters accessible in this measurement.

        """
        return [self.__amp, self.__freq, self.__time]

    def measure_normalised_scalar(self):
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
        t = self.__time.get_value()
        diff_sq = (q_freq - freq) ** 2
        return jnp.abs(jnp.cos(jnp.sqrt(diff_sq + amp**2) / 2 * t) / jnp.sqrt(1 + diff_sq / (amp**2))) ** 2
