"""Testing the Rabi Analytic Model."""

import pytest
from numpy.testing import assert_almost_equal

from paraqeet.optimization_map import OptimizationMap
from paraqeet.measurement.rabi_experiment import RabiExperiment
from paraqeet.optimizers.scipy_optimizer import ScipyOptimizer


FREQ = 4.8e9
RABI_NAME = "Analytic Rabi Model"


@pytest.fixture
def opt(rabi):
    """Create optimization map with Rabi model."""
    optmap = OptimizationMap()
    optmap.add(rabi, rabi.get_parameters())
    return ScipyOptimizer(rabi, optmap)


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
    res = opt.optimize()
    assert res.value < 1e-8


def test_find_resonance(rabi, opt) -> None:
    """Check for resonance."""
    opt.optimize()
    params = rabi.get_parameters()
    assert_almost_equal(FREQ / 1e9, params[1].get_value() / 1e9, decimal=4)
