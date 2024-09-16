"""Test the mixed state transfer fidelity."""

import numpy as np
import pytest

from cthree.measurement.MixedStateTransferFidelity import (
    MixedStateTransferFidelity,
)
from test.propagation.IdentityPropagation import IdentityPropagation
from test.propagation.RandomPropagation import RandomPropagation


def randomMixedState(dimension):
    """Generate random mixed states."""
    state = np.random.random((dimension, dimension)) + 1j * np.random.random(
        (dimension, dimension)
    )
    state = state @ np.conjugate(state.T)
    return state / np.trace(state)


def test_limits_vectors():
    """Test state vector limits.

    The fidelity for state vectors is always in the interval [0, 1).

    """
    for size in range(2, 30):
        targetState = randomMixedState(size)
        propagation = RandomPropagation(size, True)
        times = np.array([1.0])
        measurement = MixedStateTransferFidelity(
            propagation, targetState, times
        )

        for i in range(100):
            m = measurement.measure()
            assert 0.0 <= m <= 1.0


def test_vector_equality():
    """Test that F(v,v) = 1 for state vectors."""
    for size in range(2, 30):
        for i in range(100):
            state = randomMixedState(size)
            propagation = IdentityPropagation()
            propagation.setInitialState(state)
            measurement = MixedStateTransferFidelity(
                propagation, state, np.array([1.0])
            )
            m = measurement.measure()
            np.testing.assert_almost_equal(m, 1.0, decimal=2)


def test_incompatible_shape():
    """Test incompatible shape of propagators.

    The propagators which have a different dimension than the
    target state should raise an exception.

    Raises
    ------
    Exception
        Different dimensions of the propagators raise an exception.

    """
    allDims = np.arange(2, 30)
    for dim in allDims:
        for i in range(100):
            targetState = randomMixedState(dim)

            # create a propagator of a different dimension
            dimensions = np.delete(allDims, np.where(allDims == dim)[0][0])
            propagation = RandomPropagation(np.random.choice(dimensions), True)

            measurement = MixedStateTransferFidelity(
                propagation, targetState, np.array([1.0])
            )
            with pytest.raises(Exception):
                measurement.measure()
