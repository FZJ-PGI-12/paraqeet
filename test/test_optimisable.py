"""Testing the Optimisables."""

import numpy as np

from cthree.optimisable import Optimisable
from cthree.quantity import Quantity


class TestOptimisable(Optimisable):
    """An optimisable implementation.

    Does nothing except providing some random parameters.

    Parameters
    ----------
    randomQuantity
        New randomly generated Quantity object.
    numParams : int
        Number of parameters.
    """

    def __init__(self, randomQuantity, numParams: int):
        super().__init__()
        self._optimisableParameters = [randomQuantity(np.random.randint(1, 20)) for i in range(numParams)]

    def get_parameters(self) -> list[Quantity]:
        """Get optimisable parameters."""
        return self._optimisableParameters
