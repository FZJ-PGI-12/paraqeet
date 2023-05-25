from abc import abstractmethod
from typing import List
from cthree.Quantity import Quantity


class Measurement:
    """
    Represents any observable and the process of measurement itself. The observable is measured after the propagation
    class has solved the equation of motion.
    """

    def __init__(self):
        pass

    @abstractmethod
    def getParameters(self) -> List[Quantity]:
        """
        Return a list of parameters accessible in this measurement.
        """
        raise NotImplementedError()

    @abstractmethod
    def measure(self) -> float:
        """
        Measures the observable and returns the value. This function must be implemented by subclasses.
        """
        raise NotImplementedError()
