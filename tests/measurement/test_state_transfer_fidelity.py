"""Test the state transfer fidelity model."""

import numpy as np
import pytest

from paraqeet.measurement.state_transfer_fidelity import (
    StateTransferFidelity,
    StateTransferFidelityAD,
)
from tests.propagation.identity_propagation import IdentityPropagation
from tests.propagation.random_propagation import RandomPropagation


@pytest.fixture
def identity_propagation():
    """Return a mock identity propagation object."""
    return IdentityPropagation()


def test_limits_vectors(random_state):
    """Test fidelity for state vectors is always in the interval [0, 1)."""
    for size in range(2, 30):
        initial_state = random_state(size)
        target_state = random_state(size)
        propagation = RandomPropagation(size, False)
        times = np.array([1.0])
        measurement = StateTransferFidelity(
            propagation,
            initial_state,
            target_state,
            times,
        )

        for _ in range(20):
            m = measurement.measure()
            assert 0.0 <= m
            m = measurement.measure_scalar()
            assert 0.0 <= m
            m = measurement.calculate_normalized_scalar()
            assert 0.0 <= m <= 1.0
            m, grad = measurement.calculate_normalized_scalar_and_gradient()
            assert 0.0 <= m


@pytest.mark.filterwarnings("ignore:Different shapes for")
def test_limit_projected_vectors(random_state):
    """Test the projection to a subspace.

    The projection to a subspace should not increase the
    range of possible measurement outcomes for state vectors.

    """
    times = np.array([1.0])
    for size in range(3, 30):
        for projectedSize in range(2, size):
            inital_state = random_state(size)
            target_state = random_state(projectedSize)
            propagation = RandomPropagation(size, False)
            measurement = StateTransferFidelity(
                propagation=propagation,
                initial_state=inital_state,
                target_state=target_state,
                times=times,
            )

            measurement.restrict_subsystems([size], [projectedSize])
            for _ in range(20):
                m = measurement.measure()
                assert 0.0 <= m <= 1.0


def test_vector_equality(identity_propagation, random_state):
    """Test that F(v,v) = 1 for state vectors."""
    for size in range(2, 30):
        for _ in range(100):
            state = random_state(size)
            identity_propagation.set_initial_state(state)
            measurement = StateTransferFidelity(identity_propagation, state, state, np.array([1.0]))
            m = measurement.measure()
            np.testing.assert_almost_equal(m, 1.0)


@pytest.mark.filterwarnings("ignore:Different shapes for")
def test_incompatible_shape(identity_propagation, random_state):
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
            initialState = random_state(dim)
            dimensions = np.delete(allDims, np.where(allDims == dim)[0][0])
            targetState = random_state(np.random.choice(dimensions))

            fid = StateTransferFidelity(identity_propagation, initialState, targetState, np.array([1.0]))

            fid_AD = StateTransferFidelityAD(identity_propagation, initialState, targetState, np.array([1.0]))

            with pytest.raises(Exception):
                fid.measure()
            with pytest.raises(Exception):
                fid_AD.measure()

