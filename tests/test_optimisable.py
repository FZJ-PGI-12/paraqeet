"""Testing the Optimisables."""

import numpy as np

from paraqeet.optimisable import Optimisable
from paraqeet.quantity import Quantity


class DummyOptimisable(Optimisable):
    """An optimisable implementation.

    Does nothing except providing some random parameters.

    Parameters
    ----------
    random_quantity: Quantity
        New randomly generated Quantity object.
    num_params: int
        Number of parameters.
    """

    def __init__(self, random_quantity, num_params: int):
        super().__init__()
        self._optimisable_parameters = [random_quantity(np.random.randint(1, 20)) for i in range(num_params)]

    def get_parameters(self) -> list[Quantity]:
        """Get optimisable parameters."""
        return self._optimisable_parameters
