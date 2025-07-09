"""Test the Euler propagation model."""

import pytest
import numpy as np

from paraqeet.exceptions import ConfigurationException
from paraqeet.propagation.euler import Euler
from test.model.dummy_model import DummyModel
from test.model.empty_hamiltonian import EmptyHamiltonian


@pytest.fixture
def euler():
    """Return a Euler propagation model generating method."""

    def _method(dimension):
        return Euler(DummyModel(EmptyHamiltonian(dimension)))

    return _method


def test_parameters(euler):
    """Test parameters of the model."""
    assert euler(2).get_parameters() == []


def test_state_dimension_vector(random_state, euler, ts):
    """Test the dimension and the norm of the state vector.

    The dimension and the norm should be the same.

    """
    for i in range(10):
        dim = np.random.randint(2, 30)
        state = random_state(dim)
        propagation = euler(dim)
        propagation.set_initial_state(state)
        propagatedStates = propagation.propagate(ts)
        assert propagatedStates.shape[0] == len(ts)
        assert propagatedStates.shape[1:] == state.shape


def test_state_dimension_matrix(random_matrix, euler, ts):
    """Test the state dimension matrix."""
    for i in range(10):
        dim = np.random.randint(2, 30)
        state = random_matrix(dim, dim)
        propagation = euler(dim)
        propagation.set_initial_state(state)
        propagatedStates = propagation.propagate(ts)
        assert propagatedStates.shape[0] == len(ts)
        assert propagatedStates.shape[1:] == state.shape


def test_needs_initial_state(random_state, euler):
    for dim in range(2, 10):
        random_time_vector = np.linspace(0.0, np.random.randint(1, 10) * np.random.rand(), np.random.randint(1, 10))
        propagation = euler(dim)
        with pytest.raises(ConfigurationException):
            propagation.propagate(random_time_vector)

        propagation.set_initial_state(random_state(dim))
        propagation.propagate(random_time_vector)
