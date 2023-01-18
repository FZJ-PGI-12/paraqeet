import numpy as np
from numpy.testing import assert_almost_equal

from cthree.measurement.RabiExperiment import RabiExperiment
from cthree.ScipyOptimiser import ScipyOptimiser


FREQ = 4.8e9 * 2 * np.pi
rabi = RabiExperiment(FREQ)
params = rabi.getParameters()
opt = ScipyOptimiser(rabi)
res = opt.optimise()


def test_rabi() -> None:
    """
    Check that the rabi optimization goes below threshold.
    """
    assert res.fun < 1e-8


def test_find_resonance() -> None:
    assert_almost_equal(1, params[1].getValue(), decimal=4)
