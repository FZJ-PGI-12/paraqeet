from typing import List, Optional
import numpy as np

from cthree.signals.Device import Device
from cthree.signals.Generator import Generator


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
            sig += dev.computeOutput(t)
        return sig
