from typing import List
import numpy as np

from cthree.Optimisable import Optimisable
from cthree.Quantity import Quantity


class TestOptimisable(Optimisable):
    """
    An optimisable implementation that does nothing except providing some random parameters.
    """
    def __init__(self, randomQuantity, numParams: int):
        super().__init__()
        self._optimisableParameters = [randomQuantity(np.random.randint(1, 20)) for i in range(numParams)]

    def getParameters(self) -> List[Quantity]:
        return self._optimisableParameters
