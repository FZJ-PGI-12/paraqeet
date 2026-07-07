"""Abstract base class for the signal generation stack that produces the control signal."""

from typing import override

from paraqeet.differentiable import Differentiable
from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Array


class Generator(Optimizable, Differentiable):
    """Marker class for signal generation stack.

    paraqeet includes a detailed simulation of the control stack.
    Each component in the stack and its functions are
    simulated individually and combined here.

    Example: A local oscillator and arbitrary waveform generator signal
    are put through via a mixer device to produce an effective modulated signal.
    """

    @override
    def get_value(self, times: Array) -> Array:
        pass
