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

    def __init__(self, devices: list | None):
        self.__devices = devices or []

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
        return sig

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
        gradients = jnp.zeros(shape=(t.shape[0], 0))
        for dev in self.__devices:
            grad = dev.computeGradient(t)
            gradients = jnp.append(gradients, grad, axis=1)
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
        gradients = jnp.zeros(shape=(0,))
        for dev in self.__devices:
            grad = jnp.squeeze(dev.computeGradient(t), axis=0)
            gradients = jnp.append(gradients, grad, axis=0)
        return jnp.array(gradients)

    def getParameters(self) -> list[Quantity]:
        """Collect and returns the parameters of all devices.

        Returns
        -------
        jax.Array
            Returns the parameters from all devices.

        """
        pars = []
        for dev in self.__devices:
            pars.extend(dev.getParameters())
        return pars
