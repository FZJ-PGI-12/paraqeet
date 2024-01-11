from abc import abstractmethod
from typing import List, Tuple

import numpy as np

from cthree.Optimisable import Optimisable
from cthree.Quantity import Quantity


class Measurement(Optimisable):
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
    def measure(self) -> np.ndarray:
        """
        Measures the observable and returns the value. This function must be implemented by subclasses.
        """
        raise NotImplementedError()

    def measureNormalised(self) -> np.ndarray:
        """
        Measures the observable and returns the value between 0 and 1, 1 representing the perfect result.
        This function must be implemented by subclasses, unless identical to self.measure().
        """
        return self.measure()

    def measureWithGradient(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the measurement value as in measureNormalised() but with the gradient wrt to parameters.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of function value and gradient of shape (n_parameters,)

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()
