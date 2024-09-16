"""Test the Runge-Kutta propagation model."""

import pytest
import numpy as np
from cthree.Exceptions import ConfigurationException
from cthree.propagation.RungeKutta import RungeKutta
from test.model.DummyModel import DummyModel
from test.model.EmptyHamiltonian import EmptyHamiltonian


@pytest.fixture
def rk():
    """Return a Runge-Kutta model generating method."""

    def _method(dimension):
        return RungeKutta(DummyModel(EmptyHamiltonian(dimension)))

    return _method


def test_parameters(rk):
    """Test parameters of the model."""
    assert rk(2).getParameters() == []


def test_state_dimension(rk, ts, randomState):
    """Test the state vector dimensions.

    Dimension and norm of state vectors should be the same after propagation.
    """
    dim = np.random.randint(1, 100)
    state = randomState(dim)
    rungeKutta = rk(dim)
    rungeKutta.setInitialState(state)
    propagatedStates = rungeKutta.propagate(ts)
    assert len(propagatedStates) == len(ts)
    assert propagatedStates[-1].shape == state.shape


def test_initial_state(rk, ts):
    """Test whether the initial state is set.

    Raises
    ------
    cthree.Exceptions.ConfigurationException
        If the initial state is not set.

    """
    rungeKutta = rk(np.random.randint(1, 100))
    with pytest.raises(
        ConfigurationException, match="Initial state is not set"
    ):
        rungeKutta.propagate(ts)


def test_time_steps(rk, randomState):
    """Test the Runge-Kutta time steps for propagation.

    Raises
    ------
    ValueError
        If the Runge-Kutta propagation does not get at least two time steps.

    """
    time = np.array([0])
    dim = np.random.randint(1, 100)
    state = randomState(dim)
    rungeKutta = rk(dim)
    rungeKutta.setInitialState(state)
    with pytest.raises(
        ValueError,
        match="Runge-Kutta propagation needs at least two time steps",
    ):
        rungeKutta.propagate(time)
