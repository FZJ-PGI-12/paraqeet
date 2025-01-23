"""Test the state transfer fidelity model."""

import numpy as np
import pytest

from cthree.measurement.StateTransferFidelity import (
    StateTransferFidelity,
    StateTransferFidelityAD,
)
from test.propagation.IdentityPropagation import IdentityPropagation
from test.propagation.RandomPropagation import RandomPropagation


@pytest.fixture
def identityPropagation():
    """Return a mock identity propagation object."""
    return IdentityPropagation()


def test_limits_vectors(randomState):
    """Test fidelity for state vectors is always in the interval [0, 1)."""
    for size in range(2, 30):
        initialState = randomState(size)
        targetState = randomState(size)
        propagation = RandomPropagation(size, False)
        times = np.array([1.0])
        measurement = StateTransferFidelity(
            propagation,
            initialState,
            targetState,
            times,
        )

        for _ in range(20):
            m = measurement.measure()
            assert 0.0 <= m <= 1.0


@pytest.mark.filterwarnings("ignore:Different shapes for")
def test_limit_projected_vectors(randomState):
    """Test the projection to a subspace.

    The projection to a subspace should not increase the
    range of possible measurement outcomes for state vectors.

    """
    times = np.array([1.0])
    for size in range(3, 30):
        for projectedSize in range(2, size):
            inital_state = randomState(size)
            target_state = randomState(projectedSize)
            propagation = RandomPropagation(size, False)
            measurement = StateTransferFidelity(
                propagation=propagation,
                initialState=inital_state,
                targetState=target_state,
                times=times,
            )

            measurement.restrict_subsystems([size], [projectedSize])
            for _ in range(20):
                m = measurement.measure()
                assert 0.0 <= m <= 1.0


def test_vector_equality(identityPropagation, randomState):
    """Test that F(v,v) = 1 for state vectors."""
    for size in range(2, 30):
        for _ in range(100):
            state = randomState(size)
            identityPropagation.set_initial_state(state)
            measurement = StateTransferFidelity(identityPropagation, state, state, np.array([1.0]))
            m = measurement.measure()
            np.testing.assert_almost_equal(m, 1.0)


@pytest.mark.filterwarnings("ignore:Different shapes for")
def test_incompatible_shape(identityPropagation, randomState):
    """Test incompatible shapes for initial and target states.

    A set of initial and target states with different dimensions
    should raise an exception.

    Raises
    ------
    Exception
        Different dimensions for the initial and target states
        raise an exception.

    """
    allDims = np.arange(2, 30)
    for dim in allDims:
        for i in range(10):
            initialState = randomState(dim)
            dimensions = np.delete(allDims, np.where(allDims == dim)[0][0])
            targetState = randomState(np.random.choice(dimensions))

            fid = StateTransferFidelity(identityPropagation, initialState, targetState, np.array([1.0]))

            fid_AD = StateTransferFidelityAD(identityPropagation, initialState, targetState, np.array([1.0]))

            with pytest.raises(Exception):
                fid.measure()
            with pytest.raises(Exception):
                fid_AD.measure()


def test_no_parameters(identityPropagation, randomState):
    """Test the no parameter case."""
    state = randomState(np.random.randint(2, 30))
    measurement = StateTransferFidelity(identityPropagation, state, state, np.array([1.0]))
    assert measurement.get_parameters() == []

    measurement = StateTransferFidelityAD(identityPropagation, state, state, np.array([1.0]))
    assert measurement.get_parameters() == []
