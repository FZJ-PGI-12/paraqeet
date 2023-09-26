import copy
from typing import Tuple

import numpy as np
import numpy.testing as testing

from cthree.Quantity import Quantity
from cthree.measurement.MixedStateTransferFidelity import MixedStateTransferFidelity
from test.IdentityPropagation import IdentityPropagation

from test.RandomPropagation import RandomPropagation


def randomMixedState(dimension):
    state = np.random.random((dimension, dimension)) + 1j * np.random.random((dimension, dimension))
    state = state @ np.conjugate(state.T)
    return state / np.trace(state)


# test that the fidelity for state vectors is always in the interval [0, 1)
def test_limits_vectors():
    for size in range(2, 30):
        state = randomMixedState(size)
        initialState = randomMixedState(size)
        targetState = randomMixedState(size)
        propagation = RandomPropagation(size, False)
        times = np.array([1.0])
        measurement = MixedStateTransferFidelity(propagation, targetState, times)

        for i in range(100):
            m = measurement.measure()
            assert 0.0 <= m <= 1.0


# test that F(v,v) = 1 for state vectors
def test_vector_equality():
    for size in range(2, 30):
        for i in range(100):
            state = randomMixedState(size)
            propagation = IdentityPropagation()
            propagation.setInitialState(state)
            measurement = MixedStateTransferFidelity(propagation, state, np.array([1.0]))
            m = measurement.measure()
            np.testing.assert_almost_equal(m, 0.0, decimal=2)
