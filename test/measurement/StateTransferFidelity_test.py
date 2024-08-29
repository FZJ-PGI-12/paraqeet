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
    return IdentityPropagation()


# test that the fidelity for state vectors is always in the interval [0, 1)
def test_limits_vectors(randomState):
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
def test_vector_equality(identityPropagation, randomState):
    for size in range(2, 30):
        for _ in range(100):
            state = randomState(size)
            identityPropagation.setInitialState(state)
            measurement = StateTransferFidelity(
                identityPropagation, state, state, np.array([1.0])
            )
            m = measurement.measure()
            np.testing.assert_almost_equal(m, 1.0)


# Test that a set of initial and target state with different dimensions raise an exception
@pytest.mark.filterwarnings("ignore:Different shapes for")
def test_incompatible_shape(identityPropagation, randomState):
    allDims = np.arange(2, 30)
    for dim in allDims:
        for i in range(10):
            initialState = randomState(dim)
            dimensions = np.delete(allDims, np.where(allDims == dim)[0][0])
            targetState = randomState(np.random.choice(dimensions))

            fid = StateTransferFidelity(
                identityPropagation, initialState, targetState, np.array([1.0])
            )

            fid_AD = StateTransferFidelityAD(
                identityPropagation, initialState, targetState, np.array([1.0])
            )

            with pytest.raises(Exception):
                fid.measure()
            with pytest.raises(Exception):
                fid_AD.measure()


def test_no_parameters(identityPropagation, randomState):
    state = randomState(np.random.randint(2, 30))
    measurement = StateTransferFidelity(
        identityPropagation, state, state, np.array([1.0])
    )
    assert measurement.getParameters() == []

    measurement = StateTransferFidelityAD(
        identityPropagation, state, state, np.array([1.0])
    )
    assert measurement.getParameters() == []

# Test that the projection to a subspace is working
def test_projection_state(identityPropagation, randomState):
    for dim in np.arange(2, 30):
        for dimProjected in np.arange(2, dim):
            for version in [StateTransferFidelity]: #StateTransferFidelityAD
                initialState = randomState(dim)
                targetState = initialState[0:dimProjected]
                times = np.linspace(0, 1.0, 100)
                measurement = version(identityPropagation, initialState, targetState, times)

                # Test the shape of the output of the preprocess function
                measurement.restrictSubsystems([dim], [dimProjected])
                #projectedState = measurement._preprocess(initialState)
                #assert projectedState.shape == targetState.shape

                # Test the result of the measurement after projection
                result = measurement.measure()
                np.testing.assert_almost_equal(result, 1.0)
