from typing import List

from Quantity import Quantity


class Optimisable:
    """
    This interface must be implemented by any class that provides optimisable parameters. The optimiser will collect
    all parameters (by reference) and update their values.
    """
    def getParameters(self) -> List[Quantity]:
        pass
