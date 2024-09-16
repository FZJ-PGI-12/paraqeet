"""Test the weighted sum goal."""

from test.propagation.IdentityPropagation import IdentityPropagation

import numpy as np
import pytest
from cthree.Exceptions import ConfigurationException
from cthree.measurement.UnitaryFidelity import UnitaryFidelity
from cthree.measurement.WeightedSumGoal import WeightedSumGoal


def test_WeightedSumGoal(randomUnitaryMatrix):
    """Test the weighted sum goal function."""
    meas = []
    for _ in range(np.random.randint(2, 10)):
        gate = randomUnitaryMatrix(np.random.randint(2, 4))
        propagation = IdentityPropagation()
        propagation.setInitialState(gate)
        meas.append(UnitaryFidelity(propagation, gate, np.array([1.0])))
    weights = np.random.random(len(meas))
    weights /= sum(weights)
    goal = WeightedSumGoal(measurements=meas, weights=weights)
    assert goal.measure() > 0


def test_WeightedSumGoalMismatchedWeights():
    """Test the mismatched weights from a weighted sum goal.

    Raises
    ------
    cthree.Exceptions.ConfigurationException
        Raise an exception with the weighted sum goal with weights.

    """
    with pytest.raises(ConfigurationException):
        WeightedSumGoal(measurements=[], weights=[0.2, 0.3, 0.5])


def test_WeightedSumGoalWeightsNotNormalised():
    """Test the not normalised weighted sum goal function.

    Raises
    ------
    cthree.Exceptions.ConfigurationException
        Raise an exception with the weighted sum goal with weights.

    """
    with pytest.raises(UserWarning):
        WeightedSumGoal(
            measurements=[None, None, None], weights=[0.4, 0.3, 0.5]
        )
