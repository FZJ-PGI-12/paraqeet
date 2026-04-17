"""Test the Scipy piecewise exponentiation GOAT solver."""

import numpy as np
import pytest

from paraqeet.model.schroedinger_equation import SchroedingerEquation
from paraqeet.propagation.scipy_expm_goat import ScipyExpmGOAT
from tests.model.empty_hamiltonian import EmptySystem
from tests.propagation.test_common_propagation import check_propagation


@pytest.fixture
def expm():
    """Return a Scipy piecewise exponentiation solver generating function."""

    def _method(dimension, resolution, initial_state):
        sys = EmptySystem(dimension)
        schreq = SchroedingerEquation(
            hamiltonian_func=sys.get_hamiltonian, hamiltonian_and_gradient_func=sys.get_hamiltonian_and_gradient
        )
        return ScipyExpmGOAT(eom_func=schreq.get_value, resolution=resolution, initial_state=initial_state)

    return _method


def test_resolution(expm, random_state):
    """Test the resolution of the solver."""
    for _ in range(10):
        dim = np.random.randint(2, 100)
        propagation = expm(dimension=dim, resolution=3, initial_state=random_state(dim))
        resolution = np.random.randint(1, 1000)
        propagation.resolution = resolution
        assert propagation.resolution == resolution


def test_state_dimension_vector(random_state, expm, ts):
    """Test dimension and norm of state vectors after propagation.

    The dimension and norm of state vectors should be the same
    after propagation.

    """
    for _ in range(10):
        dim = np.random.randint(2, 10)
        state = random_state(dim)
        propagation = expm(dim, resolution=3, initial_state=state)
        propagated_states = propagation.propagate(ts)
        assert propagated_states.shape[0] == len(ts)
        assert propagated_states.shape[1:] == state.shape
