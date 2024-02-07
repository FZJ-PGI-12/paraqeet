import numpy as np
import pytest

from cthree.measurement.StateTransferFidelity import StateTransferFidelity
from test.propagation.IdentityPropagation import IdentityPropagation
from test.propagation.RandomPropagation import RandomPropagation


def randomState(dimension):
    state = np.random.random(dimension) + 1j * np.random.random(dimension)
    return state / np.sqrt(np.vdot(state, state))


# test that the fidelity for state vectors is always in the interval [0, 1)
def test_limits_vectors():
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

        for _ in range(100):
            m = measurement.measure()
            assert 0.0 <= m <= 1.0


# test that F(v,v) = 1 for state vectors
def test_vector_equality():
    for size in range(2, 30):
        for _ in range(100):
            state = randomState(size)
            propagation = IdentityPropagation()
            propagation.setInitialState(state)
            measurement = StateTransferFidelity(
                propagation, state, state, np.array([1.0])
            )
            m = measurement.measure()
            np.testing.assert_almost_equal(m, 1.0)


# Test that a set of initial and target state with different dimensions raise an exception
def test_incompatible_shape():
    allDims = np.arange(2, 30)
    for dim in allDims:
        for i in range(100):
            initialState = randomState(dim)
            dimensions = np.delete(allDims, np.where(allDims == dim)[0][0])
            targetState = randomState(np.random.choice(dimensions))

            propagation = IdentityPropagation()

            with pytest.raises(Exception):
                StateTransferFidelity(
                    propagation, initialState, targetState, np.array([1.0])
                )
