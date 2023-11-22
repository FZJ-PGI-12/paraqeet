"""
Signal generation stack.

Contrary to most quantum simulators, C^3 includes a detailed simulation of the control
stack. Each component in the stack and its functions are simulated individually and
combined here.

Example: A local oscillator and arbitrary waveform generator signal
are put through via a mixer device to produce an effective modulated signal.
"""

from abc import abstractmethod
from typing import List, Dict
from cthree.Optimisable import Optimisable
from cthree.signal.Device import Device


class Generator(Optimisable):
    """
    TODO: Copy as before
    """

    __chains: Dict[str, Dict[str, List[str]]] = {}
    __devices: List[Device]

    @abstractmethod
    def generateSignal(self, instr):
        raise NotImplementedError()

    @abstractmethod
    def generateSignalGradient(self, instr):
        raise NotImplementedError()
