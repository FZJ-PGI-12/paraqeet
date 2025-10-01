"""Testing the Rabi Analytic Model."""

import pytest
from numpy.testing import assert_almost_equal

from paraqeet.optimisation_map import OptimisationMap
from paraqeet.measurement.rabi_experiment import RabiExperiment
from paraqeet.optimisers.scipy_optimiser import ScipyOptimiser


FREQ = 4.8e9
RABI_NAME = "Analytic Rabi Model"


@pytest.fixture
def opt(rabi):
    """Create optimisation map with Rabi model."""
    optmap = OptimisationMap()
    optmap.add(rabi, rabi.get_parameters())
    return ScipyOptimiser(rabi, optmap)


@pytest.fixture
def rabi():
    """Create Rabi test object."""
    exp = RabiExperiment(FREQ)
    exp.name = RABI_NAME
    return exp


def test_name(rabi):
    """Check that the name is 'RABI_NAME'."""
    assert rabi.name == RABI_NAME


def test_rabi(opt) -> None:
    """Check that the rabi optimization goes below threshold."""
    res = opt.optimise()
    assert res.value < 1e-8


def test_find_resonance(rabi, opt) -> None:
    """Check for resonance."""
    opt.optimise()
    params = rabi.get_parameters()
    assert_almost_equal(FREQ / 1e9, params[1].get_value() / 1e9, decimal=4)
