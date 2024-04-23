import pytest
from numpy.testing import assert_almost_equal

from cthree.OptimisationMap import OptimisationMap
from cthree.measurement.RabiExperiment import RabiExperiment
from cthree.optimisers.ScipyOptimiser import ScipyOptimiser


FREQ = 4.8e9
RABI_NAME = "Analytic Rabi Model"


@pytest.fixture
def opt(rabi):
    optmap = OptimisationMap()
    optmap.add(rabi, rabi.getParameters())
    return ScipyOptimiser(rabi, optmap)


@pytest.fixture
def rabi():
    exp = RabiExperiment(FREQ)
    exp.setName(RABI_NAME)
    return exp


def test_name(rabi):
    assert rabi.getName() == RABI_NAME


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
