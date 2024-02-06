import pytest
import numpy as np
from cthree.propagation.RungeKutta import RungeKutta
from test.model.DummyModel import DummyModel
from test.model.EmptyHamiltonian import EmptyHamiltonian


@pytest.fixture
def rk():
    def _method(dimension):
        return RungeKutta(DummyModel(EmptyHamiltonian(dimension)))
    return _method


def test_parameters(rk):
    assert rk(2).getParameters() == []


# test that the dimension and norm of state vectors is the same after propagation
def test_state_dimension(randomState, rk, ts):
    for i in range(10):
        dim = np.random.randint(2, 30)
        state = randomState(dim)
        propagation = rk(dim)
        propagation.setInitialState(state)
        propagatedStates = propagation.propagate(ts)
        assert propagatedStates.shape[0] == len(ts)
        assert propagatedStates.shape[1:] == state.shape
