from typing import List

import numpy as np
import jax.numpy as jnp
from jax import Array

from cthree.Quantity import Quantity
from cthree.signal.Device import Device
from cthree.signal.Generator import Generator


class CosGenerator(Generator):
    """
    Simple sinusodial signal generation.
    """

    __devices: List[Device]

    def __init__(self, devices: List | None):
        self.__devices = devices or []

    def generateSignal(self, t: np.ndarray) -> Array:
        """
        Generate a signal for time(s) t.
        """
        sig = jnp.zeros_like(t)
        for dev in self.__devices:
            sig += jnp.reshape(dev.computeOutput(t), sig.shape)
        return sig

    def generateSignalGradient(self, t) -> Array:
        """
        Collects and returns the gradients from all devices.
        """
        gradients = jnp.zeros(shape=(t.shape[0], 0))
        for dev in self.__devices:
            grad = dev.computeGradient(t)
            gradients = jnp.append(gradients, grad, axis=1)
        return gradients

    def generateSignalGradientOneTime(self, t) -> Array:
        """
        Collects and returns the gradients from all devices.
        """
        gradients = jnp.zeros(shape=(0,))
        for dev in self.__devices:
            grad = jnp.squeeze(dev.computeGradient(t), axis=0)
            gradients = jnp.append(gradients, grad, axis=0)
        return jnp.array(gradients)

    def getParameters(self) -> List[Quantity]:
        """
        Collects and returns the parameters of all devices.
        """
        pars = []
        for dev in self.__devices:
            pars.extend(dev.getParameters())
        return pars
