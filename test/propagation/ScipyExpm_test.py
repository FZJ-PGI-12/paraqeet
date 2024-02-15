import numpy as np
import pytest
from cthree.Exceptions import ConfigurationException

from cthree.propagation.ScipyExpm import ScipyExpm
from test.model.DummyModel import DummyModel
from test.model.EmptyHamiltonian import EmptyHamiltonian


@pytest.fixture
def expm():
    def _method(dimension, res):
        return ScipyExpm(DummyModel(EmptyHamiltonian(dimension)), res=res)

    return _method


def test_parameters(expm):
    propagation = expm(dimension=np.random.randint(10), res=3)
    assert propagation.getParameters() == []


def test_resolution(expm):
    for i in range(10):
        propagation = expm(dimension=np.random.randint(2, 100), res=3)
        resolution = np.random.randint(1, 1000)
        propagation.setResolution(resolution)
        assert propagation.getResolution() == resolution


# test that the dimension and norm of state vectors is the same after propagation
def test_state_dimension_vector(randomState, expm, ts):
    for i in range(10):
        dim = np.random.randint(2, 30)
        state = randomState(dim)
        propagation = expm(dim, res=3)
        propagation.setInitialState(state)
        propagatedStates = propagation.propagate(ts)
        assert propagatedStates.shape[0] == len(ts)
        assert propagatedStates.shape[1:] == state.shape


def test_state_dimension_matrix(randomMatrix, expm, ts):
    for i in range(10):
        basis = np.random.randint(2, 30)
        dim = basis + np.random.randint(1, 3)
        state = randomMatrix(dim, basis)  # rect matrix with dim>basis
        propagation = expm(dim, res=3)
        propagation.setInitialState(state)
        propagatedStates = propagation.propagate(ts)
        assert propagatedStates.shape[0] == len(ts)
        assert propagatedStates.shape[1:] == state.shape


def test_initial_state(model):
    propagation = ScipyExpm(model=model, res=3)
    ts = np.linspace(0.0, 1e-9, 3)
    with pytest.raises(ConfigurationException, match="Initial state is not set"):
        propagation.propagate(ts)


def test_construct_times(model):
    res = 100e9

    propagation = ScipyExpm(model=model, res=res)

    # Test if times array is constructed correctly for a 1ns list
    t_start = 1e-9
    t_final = 2e-9

    time = np.array([t_start, t_final])
    steps = int(np.ceil((t_final - t_start) * res))
    full_times = np.linspace(t_start, t_final, steps, endpoint=False)
    times, dt = propagation._constructTimes(time, 1)

    assert np.all(times == full_times)
    assert np.isclose(dt, 1 / res)

    # Test if times array is constructed correctly if steps < 2
    t_start = 1e-9
    t_final = t_start + (1 / res)

    time = np.array([t_start, t_final])
    steps = int(np.ceil((t_final - t_start) * res))
    full_times = np.linspace(t_start, t_final, steps, endpoint=False)
    times, dt = propagation._constructTimes(time, 1)

    assert np.all(times == full_times)
    assert np.isclose(dt, 1 / res)
