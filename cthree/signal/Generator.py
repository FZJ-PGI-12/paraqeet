from abc import abstractmethod
from typing import List, Dict

import numpy as np

from cthree.Optimisable import Optimisable
from cthree.signal.Device import Device


class Generator(Optimisable):
    """
    Signal generation stack.

    Contrary to most quantum simulators, C^3 includes a detailed simulation of the control
    stack. Each component in the stack and its functions are simulated individually and
    combined here.

    Example: A local oscillator and arbitrary waveform generator signal
    are put through via a mixer device to produce an effective modulated signal.
    """
    __chains: Dict[str, Dict[str, List[str]]] = {}
    __devices: List[Device]

    @abstractmethod
    def generateSignal(self, times: np.ndarray) -> np.ndarray:
        """
        Returns an array with the scalar signal value for each time step.
        """
        raise NotImplementedError()

    @abstractmethod
    def generateSignalGradient(self, times: np.ndarray) -> np.ndarray:
        """
        Returns an array with the gradient of the signal value for each time step. The result has the shape (t,p) where
        t is the time and p is the parameter index.
        """
        raise NotImplementedError()
