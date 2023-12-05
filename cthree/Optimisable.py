from abc import abstractmethod
from typing import List

from cthree.Quantity import Quantity


class Optimisable:
    """
    This interface must be implemented by any class that provides optimisable parameters. The optimiser will collect
    all parameters (by reference) and update their values.
    """
    _optimisableParameters = []

    @abstractmethod
    def getParameters(self) -> List[Quantity]:
        """
        Returns all parameters of this class that can be optimised.
        """
        raise NotImplementedError()

    def setOptimisableParameters(self, params: List[Quantity]) -> None:
        """
        Sets which parameters shall be considered during optimisation. All quantities that are not in the response of
        getParameters will be filtered out. This function is called by the optimised before gradient based optimisation
        to tell the layers which gradients to compute.
        """
        allParams = self.getParameters()
        self._optimisableParameters = [p for p in params if p in allParams]
