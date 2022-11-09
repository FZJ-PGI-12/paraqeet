import numpy as np

from cthree.measurement.Measurement import Measurement
from cthree.Optimisable import Optimisable
from cthree.Optimiser import Optimiser


class Vector(Optimisable):
    def __init__(self, values):
        self.values = values

    def getParameters(self):
        return self.values


class SumUp(Measurement):
    """
    Example measurement that sums up a vector.
    """
    def measure(self, params) -> float:
        return np.sum(params**2)


vec = Vector(np.array([1, 1, 1, 1]))
sumup = SumUp()
opt = Optimiser(sumup, vec)


def test_scipy() -> None:
    """
    Check that the silly optimization succedes and that the final value is below threshold.
    """
    res = opt.optimise()
    assert res.success is True
    assert res.fun < 1e-12
