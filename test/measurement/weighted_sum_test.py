"""Test the weighted sum goal."""

from test.propagation.identity_propagation import IdentityPropagation

import numpy as np
import pytest
from cthree.exceptions import ConfigurationException
from cthree.measurement.unitary_fidelity import UnitaryFidelity
from cthree.measurement.weighted_sum_goal import WeightedSumGoal


def test_weighted_sum_goal(random_unitary_matrix):
    """Test the weighted sum goal function."""
    meas = []
    for _ in range(np.random.randint(2, 10)):
        gate = random_unitary_matrix(np.random.randint(2, 4))
        propagation = IdentityPropagation()
        propagation.set_initial_state(gate)
        meas.append(UnitaryFidelity(propagation, gate, np.array([1.0])))
    weights = np.random.random(len(meas))
    weights /= sum(weights)
    goal = WeightedSumGoal(measurements=meas, weights=weights)
    assert goal.measure() > 0


def test_weighted_sum_goal_mismatched_weights():
    """Test the mismatched weights from a weighted sum goal.

    Raises
    ------
    cthree.Exceptions.ConfigurationException
        Raise an exception with the weighted sum goal with weights.

    """
    with pytest.raises(ConfigurationException):
        WeightedSumGoal(measurements=[], weights=[0.2, 0.3, 0.5])


def test_weighted_sum_goal_weights_not_normalised():
    """Test the not normalised weighted sum goal function.

    Raises
    ------
    cthree.Exceptions.ConfigurationException
        Raise an exception with the weighted sum goal with weights.

    """
    with pytest.raises(UserWarning):
        WeightedSumGoal(measurements=[None, None, None], weights=[0.4, 0.3, 0.5])
