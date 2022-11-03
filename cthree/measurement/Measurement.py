import numpy as np

from cthree.propagation.Propagation import Propagation


class Measurement:
    """
    Represents any observable and the process of measurement itself. The observable is measured after the propagation
    class has solved the equation of motion.
    """
    __propagation: Propagation

    def __init__(self, propagation: Propagation):
        self.__propagation = propagation

    def measure(self) -> float:
        """
        Measures the observable and returns the value. This function must be implemented by subclasses.
        """
        pass

    def _getPropagator(self) -> np.array:
        """
        Provides the propagator to implementing classes.
        """
        return self.__propagation.propagate()
