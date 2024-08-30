"""
Tests for the utility functions in the abstract Measurement class. These tests should be independent of specific imp
"""
import numpy as np
import pytest

from test.measurement.RandomMeasurement import RandomMeasurement
from test.propagation.RandomPropagation import RandomPropagation


def randomState(dimension):
    state = np.random.random((dimension, 1)) + 1j * np.random.random((dimension, 1))
    return state / np.sqrt(np.vdot(state, state))


# Test that projection to a subspace does not increase the range of possible measurement outcomes for state vectors
@pytest.mark.filterwarnings("ignore:Different shapes for")
def test_limit_projected_vectors(randomState):
    times = np.array([1.0])
    for size in range(3, 30):
        for projectedSize in range(2, size):
            propagation = RandomPropagation(size, False)
            measurement = RandomMeasurement(propagation=propagation, times=times)
            measurement.restrictSubsystems([size], [projectedSize])
            for _ in range(20):
                m = measurement.measure()
                assert 0.0 <= m <= 1.0


# Test that projection to a subspace does not increase the range of possible measurement outcomes for propagators
def test_gate_shape(randomUnitaryMatrix):
    times = np.array([1.0])
    for size in range(5, 30):
        for projectedSize in range(2, size):
            propagation = RandomPropagation(size, True)
            measurement = RandomMeasurement(propagation=propagation, times=times)
            measurement.restrictSubsystems([size], [projectedSize])
            for _ in range(20):
                m = measurement.measure()
                assert 0.0 <= m <= 1.0
