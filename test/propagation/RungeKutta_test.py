import pytest
import numpy as np
from cthree.Exceptions import ConfigurationException
from cthree.propagation.RungeKutta import RungeKutta
from test.conftest import DIMS


@pytest.fixture
def rk():
    def _method(dimension):
        return RungeKutta(DummyModel(EmptyHamiltonian(dimension)))
    return _method


def test_parameters(rk):
    assert rk(2).getParameters() == []


# test that the dimension and norm of state vectors is the same after propagation
def test_state_dimension(rk, ts, randomState):
    state = randomState(DIMS)
    rk.setInitialState(state)
    propagatedStates = rk.propagate(ts)
    assert len(propagatedStates) == len(ts)
    assert propagatedStates[-1].shape == state.shape


def test_initial_state(rk, ts):
    with pytest.raises(ConfigurationException, match="Initial state is not set"):
        rk.propagate(ts)


def test_time_steps(rk, randomState):
    time = np.array([0])
    state = randomState(DIMS)
    rk.setInitialState(state)
    with pytest.raises(
        ValueError, match="Runge-Kutta propagation needs at least two time steps"
    ):
        rk.propagate(time)
