"""Test the Runge-Kutta propagation model."""

import numpy as np
import pytest

from paraqeet.model.schroedinger_equation import SchroedingerEquation
from paraqeet.propagation.runge_kutta import RungeKutta
from tests.model.empty_hamiltonian import EmptySystem
from tests.propagation.test_common_propagation import check_propagation


@pytest.fixture
def rk():
    """Return a Runge-Kutta model generating method."""

    def _method(dimension):
        sys = EmptySystem(dimension)
        eom = SchroedingerEquation(
            hamiltonian_func=sys.get_value,
            hamiltonian_and_gradient_func=sys.get_value_and_gradient,
        )
        return RungeKutta(eom_func=eom.get_value, resolution=1e9, initial_state=np.array([[1.0], [0.0j]]))

    return _method


def test_state_dimension(rk, ts, random_state):
    """Test the state vector dimensions.

    Dimension and norm of state vectors should be the same after propagation.
    """
    dim = np.random.randint(1, 100)
    state = random_state(dim)
    runge_kutta = rk(dim)
    runge_kutta.initial_state = state
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
        check_propagation(propagation, dim)


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
    runge_kutta.initial_state = state
    with pytest.raises(
        ValueError,
        match="RungeKutta.propagate needs at least two time steps",
    ):
        runge_kutta.propagate(time)
