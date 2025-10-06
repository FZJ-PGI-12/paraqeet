"""Testing the Optimizables."""

import numpy as np

from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Quantity


class DummyOptimizable(Optimizable):
    """An optimizable implementation.

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
        self._optimizable_parameters = [random_quantity(np.random.randint(1, 20)) for i in range(num_params)]

    def get_parameters(self) -> list[Quantity]:
        """Get optimizable parameters."""
        return self._optimizable_parameters
