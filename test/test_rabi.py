import pytest
from numpy.testing import assert_almost_equal

from cthree.measurement.RabiExperiment import RabiExperiment
from cthree.ScipyOptimiser import ScipyOptimiser


FREQ = 4.8e9


@pytest.fixture
def opt(rabi):
    return ScipyOptimiser(rabi)


@pytest.fixture
def rabi():
    return RabiExperiment(FREQ)


def test_rabi(opt) -> None:
    """
    Check that the rabi optimization goes below threshold.
    """
    res = opt.optimise()
    assert res.fun < 1e-8


def test_find_resonance(rabi, opt) -> None:
    opt.optimise()
    params = rabi.getParameters()
    assert_almost_equal(FREQ / 1e9, params[1].getValue() / 1e9, decimal=4)
