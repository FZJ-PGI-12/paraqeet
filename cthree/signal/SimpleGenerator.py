from typing import List, Optional
import numpy as np
from cthree.Quantity import Quantity

from cthree.signal.Device import Device
from cthree.signal.Generator import Generator


class CosGenerator(Generator):
    """
    Simple sinusodial signal generation.
    """

    __devices: List[Device]

    def __init__(self, devices: Optional[List]):
        self.__devices = devices or []

    def generateSignal(self, t):
        """
        Generate a signal for time(s) t.
        """
        sig = np.zeros_like(t)
        for dev in self.__devices:
            sig += np.reshape(dev.computeOutput(t), sig.shape)
        return sig

    def generateSignalGradient(self, instr):
        raise NotImplementedError()

    def getParameters(self) -> List[Quantity]:
        pars = []
        for dev in self.__devices:
            pars.extend(dev.getParameters())
        return pars
