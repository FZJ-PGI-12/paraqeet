"""Testing the Optimizables."""

import numpy as np

from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Quantity


class DummyOptimizable(Optimizable):
    """An optimizable implementation.

    Does nothing except providing some random parameters.
    """

    def __init__(self, random_quantity, num_params: int):
        """
        Args:
            random_quantity: New randomly generated Quantity object.
            num_params: Number of parameters.
        """
        super().__init__()
        self._optimizable_parameters = [random_quantity(np.random.randint(1, 20)) for _ in range(num_params)]

    def get_parameters(self) -> list[Quantity]:
        return self._optimizable_parameters
