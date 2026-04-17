"""Test the Euler propagation model."""

import numpy as np
import pytest

from paraqeet.model.schroedinger_equation import SchroedingerEquation
from paraqeet.propagation.euler import Euler
from tests.model.empty_hamiltonian import EmptySystem
from tests.propagation.test_common_propagation import check_propagation


@pytest.fixture
def euler():
    """Return a Euler propagation model generating method."""

    def _method(dimension):
        return Euler(SchroedingerEquation(EmptySystem(dimension)))

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
    for _ in range(10):
        dim = np.random.randint(2, 30)
        state = random_matrix(dim, dim)
        propagation = euler(dim)
        propagation.set_initial_state(state)
        propagatedStates = propagation.propagate(ts)
        assert propagatedStates.shape[0] == len(ts)
        assert propagatedStates.shape[1:] == state.shape


def test_needs_initial_state(random_state, euler):
    for dim in range(2, 10):
        propagation = euler(dim)
        check_propagation(propagation, dim)
