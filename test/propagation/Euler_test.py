import pytest
import numpy as np

from cthree.propagation.Euler import Euler
from test.model.DummyModel import DummyModel
from test.model.EmptyHamiltonian import EmptyHamiltonian


@pytest.fixture
def euler():
    def _method(dimension):
        return Euler(DummyModel(EmptyHamiltonian(dimension)))
    return _method


def test_parameters(euler):
    assert euler(2).getParameters() == []


# test that the dimension and norm of state vectors is the same after propagation
def test_state_dimension_vector(randomState, euler, ts):
    for i in range(10):
        dim = np.random.randint(2, 30)
        state = randomState(dim)
        propagation = euler(dim)
        propagation.setInitialState(state)
        propagatedStates = propagation.propagate(ts)
        assert propagatedStates.shape[0] == len(ts)
        assert propagatedStates.shape[1:] == state.shape


def test_state_dimension_matrix(randomMatrix, euler, ts):
    for i in range(10):
        dim = np.random.randint(2, 30)
        state = randomMatrix(dim, dim)
        propagation = euler(dim)
        propagation.setInitialState(state)
        propagatedStates = propagation.propagate(ts)
        assert propagatedStates.shape[0] == len(ts)
        assert propagatedStates.shape[1:] == state.shape