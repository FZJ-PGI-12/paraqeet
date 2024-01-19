from typing import List
import numpy as np
from cthree.Quantity import Quantity

from cthree.signal.Device import Device
from cthree.signal.Generator import Generator

import jax.numpy as jnp
from jax import jit
from functools import partial
from jax.typing import ArrayLike


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
        sig = np.zeros_like(t)
        for dev in self.__devices:
            sig += np.reshape(dev.computeOutput(t), sig.shape)
        return sig

    def generateSignalGradient(self, t):
        return self.__devices[0].computeGradient(t)

    def getParameters(self) -> List[Quantity]:
        pars = []
        for dev in self.__devices:
            pars.extend(dev.getParameters())
        return pars


class CosGeneratorAD(CosGenerator):
    """
    JAX numpy based CosGenerator
    """

    def __init__(self, devices: List | None):
        super().__init__(devices)
        self.__devices = devices or []

    @partial(jit, static_argnums=(0,))
    def generateSignal(self, t: ArrayLike):
        """
        Overwrite the `generateSignal` method with jnp
        """
        sig = jnp.zeros_like(t)
        for dev in self.__devices:
            sig += jnp.reshape(dev.computeOutput(t), sig.shape)
        return sig
