"""Test the weighted sum goal."""

import itertools

import numpy as np
import pytest

from paraqeet.differentiable import Differentiable
from paraqeet.exceptions import ConfigurationException
from paraqeet.measurement.unitary_fidelity import UnitaryFidelity
from paraqeet.measurement.weighted_sum_goal import WeightedSumGoal
from paraqeet.quantity import Array
from tests.measurement.constant_measurement import ConstantMeasurement
from tests.propagation.identity_propagation import IdentityPropagation


@pytest.fixture
def random_meas(random_unitary_matrix):
    meas_list = []
    num_meas = 10
    for _ in range(num_meas):
        dim = 4
        gate = random_unitary_matrix(dim)
        propagation = IdentityPropagation()
        propagation.set_initial_state(gate)
        cz = np.identity(4)
        cz[-1, -1] = -1.0
        meas_list.append(UnitaryFidelity(propagation, cz))
    return meas_list


def test_weighted_sum_goal(random_meas):
    """Test the weighted sum goal function."""
    weights = np.random.random(len(random_meas))
    weights /= sum(weights)
    goal = WeightedSumGoal(measurements=random_meas, weights=weights)
    times=np.array([1.0])
    assert goal.measure(times=times) >= 0
    # Rounding errors might cause the value to be slightly larger than 1
    assert 0 <= np.round(goal.calculate_normalized_scalar(times=times), 8) <= 1


def test_weighted_sum_goal_options(random_meas):
    """Test that the key error on the options is catched correctly"""
    weights = np.random.random(len(random_meas))
    weights /= sum(weights)
    meas_bool = [True for _ in range(len(random_meas))]
    options = {"value": 1.0, "meas_bool": meas_bool}
    with pytest.raises(KeyError, match=r".* weight and meas_bool ."):
        WeightedSumGoal(measurements=random_meas, weights=weights, sum_of_squares_options=options)
    options = {"weight": 1.0, "meas_list": meas_bool}
    with pytest.raises(KeyError, match=r".* weight and meas_bool ."):
        WeightedSumGoal(measurements=random_meas, weights=weights, sum_of_squares_options=options)


def test_weighted_sum_goal_sum_of_squares(random_meas):
    """Test the correct calculation of the weighted sum goal function"""
    weights = np.random.random(len(random_meas))
    weight_sum_of_squares = -0.05
    total_sum_of_weights = np.sum(weights) + weight_sum_of_squares
    weights = weights / total_sum_of_weights
    weight_sum_of_squares = weight_sum_of_squares / total_sum_of_weights
    meas_list_bool = [True for _ in range(len(random_meas))]
    meas_list_bool[-2] = False
    meas_list_bool[-1] = False
    sum_of_squares_options = {"weight": weight_sum_of_squares, "meas_bool": meas_list_bool}
    goal = WeightedSumGoal(measurements=random_meas, weights=weights, sum_of_squares_options=sum_of_squares_options)
    times = np.array([1.0])
    values = [m.measure(times=times) for m in random_meas]
    sum_meas: Array | float = 0.0
    for ii, w in enumerate(weights):
        sum_meas += w * values[ii]
    sum_square_diff: Array | float = 0.0
    values_in_sum_of_squares = [val for val, flag in zip(values, sum_of_squares_options["meas_bool"]) if flag]
    for meas_a, meas_b in itertools.combinations(values_in_sum_of_squares, 2):
        sum_square_diff += (meas_a - meas_b) ** 2
    sum_meas += sum_of_squares_options["weight"] * sum_square_diff
    assert np.abs(goal.measure(times=times) - sum_meas) <= 1e-8


def test_weighted_sum_goal_sum_of_squares_gradient():
    """Test the gradient only for the sum of squares
    part of the cost function
    """
    num_meas = 10
    weights = np.zeros(num_meas)
    weight_sum_of_squares = 1.0
    meas_list = []
    for _ in range(num_meas):
        value = np.random.rand()
        meas_list.append(ConstantMeasurement(propagation=IdentityPropagation(), value=value))
    meas_list_bool = [True for _ in range(num_meas)]
    sum_of_squares_options = {"weight": weight_sum_of_squares, "meas_bool": meas_list_bool}
    goal = WeightedSumGoal(measurements=meas_list, weights=weights, sum_of_squares_options=sum_of_squares_options)
    times = np.array([1.0])
    _, grad = goal.value_and_gradient(times=times)
    values_and_gradients = [m.value_and_gradient(times=times) for m in meas_list if isinstance(m, Differentiable)]
    sum_grads = np.zeros_like(values_and_gradients[0][1])
    for ii, w in enumerate(weights):
        sum_grads += w * values_and_gradients[ii][1]
    values_and_gradients_in_sum_of_squares = [
        val_grad for val_grad, flag in zip(values_and_gradients, sum_of_squares_options["meas_bool"]) if flag
    ]
    grads_diff = np.zeros_like(values_and_gradients_in_sum_of_squares[0][1])

    for meas_a, meas_b in itertools.combinations(values_and_gradients_in_sum_of_squares, 2):
        grads_diff += 2 * (meas_a[0] - meas_b[0]) * (meas_a[1] - meas_b[1])
    sum_grads += sum_of_squares_options["weight"] * grads_diff
    assert np.abs(np.sum(sum_grads - grad)) < 1e-9


def test_weighted_sum_goal_mismatched_weights():
    """Test the mismatched weights from a weighted sum goal.

    Raises
    ------
    paraqeet.Exceptions.ConfigurationException
        Raise an exception with the weighted sum goal with weights.

    """
    with pytest.raises(ConfigurationException):
        WeightedSumGoal(measurements=[], weights=[0.2, 0.3, 0.5])


def test_weighted_sum_goal_weights_not_normalized():
    """Test the not normalized weighted sum goal function.

    Raises
    ------
    paraqeet.Exceptions.ConfigurationException
        Raise an exception with the weighted sum goal with weights.

    """
    with pytest.raises(UserWarning):
        WeightedSumGoal(measurements=[None, None, None], weights=[0.4, 0.3, 0.5])
