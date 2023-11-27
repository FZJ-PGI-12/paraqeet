import numpy as np

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

        for i in range(100):
            m = measurement.measure()
            assert 0.0 <= m <= 1.0


# test that F(v,v) = 1 for state vectors
def test_vector_equality():
    for size in range(2, 30):
        for i in range(100):
            state = randomState(size)
            propagation = IdentityPropagation()
            propagation.setInitialState(state)
            measurement = StateTransferFidelity(
                propagation, state, state, np.array([1.0])
            )
            m = measurement.measure()
            np.testing.assert_almost_equal(m, 1.0)


# test that F(v,u) <= 1 for state vectors
def test_vector_normalised():
    for size in range(2, 30):
        for _ in range(100):
            state1 = randomState(size)
            state2 = randomState(size)
            propagation = IdentityPropagation()
            propagation.setInitialState(state1)
            measurement = StateTransferFidelity(
                propagation, state1, state2, np.array([1.0])
            )
            m = measurement.measure()
            np.testing.assert_array_less(m, 1.0)
            np.testing.assert_array_less(0.0, m)
