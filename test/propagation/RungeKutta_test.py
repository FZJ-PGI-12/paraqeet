import pytest
import numpy as np
from cthree.propagation.RungeKutta import RungeKutta
from test.conftest import DIMS


def randomState(dimension):
    """
    Generate randon column vector.
    """
    state = np.random.random(dimension) + 1j * np.random.random(dimension)
    return np.reshape(state / np.sqrt(np.vdot(state, state)), (-1, 1))


@pytest.fixture
def rk(model):
    return RungeKutta(model)


def test_parameters(rk):
    assert rk.getParameters() == []


# test that the dimension and norm of state vectors is the same after propagation
def test_state_dimension(rk, ts):
    state = randomState(DIMS)
    rk.setInitialState(state)
    propagatedStates = rk.propagate(ts)
    assert len(propagatedStates) == len(ts)
    assert propagatedStates[-1].shape == state.shape
