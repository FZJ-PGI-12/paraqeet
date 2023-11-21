from abc import abstractmethod
from typing import List

from cthree.Quantity import Quantity


class Optimisable:
    """
    This interface must be implemented by any class that provides optimisable parameters. The optimiser will collect
    all parameters (by reference) and update their values.
    """

    @abstractmethod
    def getParameters(self) -> List[Quantity]:
        raise NotImplementedError()
