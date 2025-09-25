"""Tests for the utility functions in the abstract Measurement class.

These tests should be independent of specific implementations.
"""

import numpy as np
import pytest

from tests.measurement.random_measurement import RandomMeasurement
from tests.propagation.random_propagation import RandomPropagation


def random_state(dimension):
    """Generate random state according to the given dimension value."""
    state = np.random.random((dimension, 1)) + 1j * np.random.random((dimension, 1))
    return state / np.sqrt(np.vdot(state, state))


@pytest.mark.filterwarnings("ignore:Different shapes for")
def test_limit_projected_vectors(random_state):
    """Test the projection to a subspace.

    The projection to a subspace should not increase the
    range of possible measurement outcomes for state vectors.

    """
    times = np.array([1.0])
    for size in range(3, 30):
        for projectedSize in range(2, size):
            propagation = RandomPropagation(size, False)
            measurement = RandomMeasurement(propagation=propagation, times=times)
            measurement.restrict_subsystems([size], [projectedSize])
            for _ in range(20):
                m = measurement.measure()
                assert 0.0 <= m <= 1.0


def test_gate_shape(random_unitary_matrix):
    """Test the gate shape.

    The projection to a subspace should not increase the
    range of possible measurement outcomes for propagators.

    """
    times = np.array([1.0])
    for size in range(5, 30):
        for projectedSize in range(2, size):
            propagation = RandomPropagation(size, True)
            measurement = RandomMeasurement(propagation=propagation, times=times)
            measurement.restrict_subsystems([size], [projectedSize])
            for _ in range(20):
                m = measurement.measure()
                assert 0.0 <= m <= 1.0
