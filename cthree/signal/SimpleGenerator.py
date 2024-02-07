from typing import List
from cthree.Quantity import Quantity

from cthree.signal.Device import Device
from cthree.signal.Generator import Generator

import jax.numpy as jnp


class CosGenerator(Generator):
    """
    Simple sinusodial signal generation.
    """

    __devices: List[Device]

    def __init__(self, devices: List | None):
        self.__devices = devices or []

    def generateSignal(self, t):
        """
        Generate a signal for time(s) t.
        """
        sig = jnp.zeros_like(t)
        for dev in self.__devices:
            sig += jnp.reshape(dev.computeOutput(t), sig.shape)
        return sig

    def generateSignalGradient(self, t):
        return self.__devices[0].computeGradient(t)

    def getParameters(self) -> List[Quantity]:
        pars = []
        for dev in self.__devices:
            pars.extend(dev.getParameters())
        return pars
