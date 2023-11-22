from abc import abstractmethod
from typing import List

import numpy as np
from cthree.Quantity import Quantity


class Measurement:
    """
    Represents any observable and the process of measurement itself. The observable is measured after the propagation
    class has solved the equation of motion.
    """

    def __init__(self, times: np.ndarray | None = None):
        self._times = times

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

    def measureNormalised(self) -> float:
        """
        Measures the observable and returns the value between 0 and 1, 1 representing the perfect result.
        This function must be implemented by subclasses, unless identical to self.measure().
        """
        return self.measure()

    def measureGradient(self) -> np.ndarray:
        """
        Compute the gradient of the measurement wrt to parameters.

        Returns
        -------
        np.ndarray
            Gradient

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()
