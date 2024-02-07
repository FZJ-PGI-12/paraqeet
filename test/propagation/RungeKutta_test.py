import pytest
import numpy as np
from cthree.Exceptions import ConfigurationException
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
def test_state_dimension(rk, ts, randomState):
    dim = np.random.randint(1, 100)
    state = randomState(dim)
    rungeKutta = rk(dim)
    rungeKutta.setInitialState(state)
    propagatedStates = rungeKutta.propagate(ts)
    assert len(propagatedStates) == len(ts)
    assert propagatedStates[-1].shape == state.shape


def test_initial_state(rk, ts):
    rungeKutta = rk(np.random.randint(1, 100))
    with pytest.raises(ConfigurationException, match="Initial state is not set"):
        rungeKutta.propagate(ts)


def test_time_steps(rk, randomState):
    time = np.array([0])
    dim = np.random.randint(1, 100)
    state = randomState(dim)
    rungeKutta = rk(dim)
    rungeKutta.setInitialState(state)
    with pytest.raises(
        ValueError, match="Runge-Kutta propagation needs at least two time steps"
    ):
        rungeKutta.propagate(time)
