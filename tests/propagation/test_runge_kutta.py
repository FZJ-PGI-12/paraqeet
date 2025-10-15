"""Test the Runge-Kutta propagation model."""

import numpy as np
import pytest

from paraqeet.propagation.runge_kutta import RungeKutta
from tests.model.dummy_model import DummyEquationsOfMotion
from tests.model.empty_hamiltonian import EmptyHamiltonian
from tests.propagation.test_common_propagation import needs_initial_state


@pytest.fixture
def rk():
    """Return a Runge-Kutta model generating method."""

    def _method(dimension):
        return RungeKutta(DummyEquationsOfMotion(EmptyHamiltonian(dimension)))

    return _method


def test_parameters(rk):
    """Test parameters of the model."""
    assert rk(2).get_parameters() == []


def test_state_dimension(rk, ts, random_state):
    """Test the state vector dimensions.

    Dimension and norm of state vectors should be the same after propagation.
    """
    dim = np.random.randint(1, 100)
    state = random_state(dim)
    runge_kutta = rk(dim)
    runge_kutta.set_initial_state(state)
    propagated_states = runge_kutta.propagate(ts)
    assert len(propagated_states) == len(ts)
    assert propagated_states[-1].shape == state.shape


def test_initial_state(rk, ts):
    """Test whether the initial state is set.

    Raises
    ------
    ConfigurationException
        If the initial state is not set.

    """
    for dim in range(2, 20):
        propagation = rk(dim)
        needs_initial_state(propagation, dim)


def test_time_steps(rk, random_state):
    """Test the Runge-Kutta time steps for propagation.

    Raises
    ------
    ValueError
        If the Runge-Kutta propagation does not get at least two time steps.

    """
    time = np.array([0])
    dim = np.random.randint(1, 100)
    state = random_state(dim)
    runge_kutta = rk(dim)
    runge_kutta.set_initial_state(state)
    with pytest.raises(
        ValueError,
        match="Runge-Kutta propagation needs at least two time steps",
    ):
        runge_kutta.propagate(time)
