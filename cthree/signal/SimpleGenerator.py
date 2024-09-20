"""Class definition for the Sinusoidal generator model."""

import numpy as np
import jax.numpy as jnp
from jax import Array

from cthree.Quantity import Quantity
from cthree.signal.Device import Device
from cthree.signal.Generator import Generator


class CosGenerator(Generator):
    """Simple sinusoidal signal generation.

    Parameters
    ----------
    devices : List[cthree.signal.Device]
        List of input devices.

    """

    __devices: list[Device]
    __phase: Quantity
    _optimisableParameters: list[Quantity] = []

    def __init__(
        self,
        devices: list[Device] | None,
        phase: Quantity | None = None,
    ):
        self.__devices = devices or []

        self.__phase = phase or Quantity(
            np.array(0.0),
            min_value=np.array(-np.pi),
            max_value=np.array(np.pi),
            unit="rad",
            name="Phase",
        )

    def getParameters(self) -> list[Quantity]:
        """Return a list of parameters.

        Collects and returns a list of parameters from the tone, generator
        and the carrier signal.

        Returns
        -------
        List[Quantity]
            All Parameters describing the signal.
        """
        pars = []
        for dev in self.__devices:
            pars += dev.getParameters()
        pars += [self.__phase]
        return pars

    def setOptimisableParameters(self, params: list[Quantity]) -> None:
        """Set specified parameters to be optimised.

        Also add the indices to `__gradArgNums` to compute the gradients.

        Parameters
        ----------
        params : list[Quantity]
        """
        super().setOptimisableParameters(params)

        for dev in self.__devices:
            dev.setOptimisableParameters(params)

    def generateSignal(self, t: np.ndarray) -> Array:
        """Generate a signal for time(s) 't'.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.Array
            Returns the signal vector.

        """
        sig = jnp.zeros_like(t)
        for dev in self.__devices:
            sig += jnp.reshape(dev.computeOutput(t), sig.shape)
        return (sig * jnp.exp(-1j * self.__phase.getValue())).real

    def generateSignalGradient(self, t) -> Array:
        """Collect and returns the gradients from all devices.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.Array
            Returns the signal gradient vector.

        """
        phase_fac_deriv = -1.0 * jnp.exp(-1j * self.__phase.getValue()).imag
        gradients = jnp.zeros(shape=(t.shape[0], 0))
        for dev in self.__devices:
            sig = dev.computeOutput(t)
            grad = dev.computeGradient(t)
            gradients = jnp.append(gradients, grad, axis=1)
            if self._isOptimised(self.__phase):
                gradients = jnp.append(
                    gradients,
                    jnp.expand_dims(
                        sig * phase_fac_deriv * self.__phase.getScale(), 1
                    ),
                    axis=1,
                )
        return gradients

    def generateSignalGradientOneTime(self, t) -> Array:
        """Return the gradients from all devices at the given time.

        Parameters
        ----------
        t : float
            Single timestamp.

        Returns
        -------
        jax.Array
            Return the gradients from all devices at one time.

        """
        phase_fac_deriv = -1 * jnp.exp(-1j * self.__phase.getValue()).imag
        gradients = jnp.zeros(shape=(0,))
        for dev in self.__devices:
            sig = dev.computeOutput(t)
            grad = jnp.squeeze(dev.computeGradient(t), axis=0)
            gradients = jnp.append(gradients, grad, axis=0)
            if self._isOptimised(self.__phase):
                gradients = jnp.append(
                    gradients,
                    sig * phase_fac_deriv * self.__phase.getScale(),
                    axis=0,
                )
        return jnp.array(gradients)
