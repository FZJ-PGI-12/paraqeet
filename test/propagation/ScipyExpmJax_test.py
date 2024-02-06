import numpy as np
import pytest

from cthree.propagation.ScipyExpmJax import ScipyExpmJax
from test.model.DummyModel import DummyModel
from test.model.EmptyHamiltonian import EmptyHamiltonian


@pytest.fixture
def expm():
    def _method(dimension, res):
        return ScipyExpmJax(DummyModel(EmptyHamiltonian(dimension)), res=res)
    return _method


def test_parameters(expm):
    propagation = expm(dimension=np.random.randint(10), res=3)
    assert propagation.getParameters() == []


def test_resolution(expm):
    for i in range(10):
        propagation = expm(dimension=np.random.randint(2, 100), res=3)
        resolution = np.random.randint(1, 1000)
        propagation.setResolution(resolution)
        assert propagation.getResolution() == resolution



# test that the dimension and norm of state vectors is the same after propagation
def test_state_dimension_vector(randomState, expm, ts):
    for i in range(10):
        dim = np.random.randint(2, 30)
        state = randomState(dim)
        propagation = expm(dim, res=3)
        propagation.setInitialState(state)
        propagatedStates = propagation.propagate(ts)
        assert propagatedStates.shape[0] == len(ts)
        assert propagatedStates.shape[1:] == state.shape


def test_state_dimension_matrix(randomMatrix, expm, ts):
    for i in range(10):
        dim = np.random.randint(2, 30)
        state = randomMatrix(dim, dim)
        propagation = expm(dim, res=3)
        propagation.setInitialState(state)
        propagatedStates = propagation.propagate(ts)
        assert propagatedStates.shape[0] == len(ts)
        assert propagatedStates.shape[1:] == state.shape