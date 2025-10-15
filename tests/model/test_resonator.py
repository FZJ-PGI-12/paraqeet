"""Test the Resonator model."""

import numpy as np
import pytest

from paraqeet.model.drive_operator import DriveOperator
from paraqeet.model.open_system import OpenSystem
from paraqeet.model.resonator import Resonator
from paraqeet.propagation.scipy_expm import ScipyExpm
from paraqeet.propagation.vern7 import Vern7
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import FlatTopGaussianEnvelope, ZeroEnvelope
from paraqeet.signal.iq_mixer import IQMixer

DIMS = 3
FREQ = 4.8e9 * 2 * np.pi
LEN_SIG = 1001
T1 = Quantity(1e-9, 1e-9, 100e-6)
TEMP = Quantity(10e-3, 1e-3, 50e-3)
T2STAR = Quantity(10e-9, 1e-9, 100e-6)


@pytest.fixture
def time_samples():
    """Generate time samples from the given signal length.

    The given signal length `LEN_SIG` is a module level global variable.

    """
    return np.linspace(0, 10e-9, LEN_SIG)


@pytest.fixture
def tone():
    """Return a object for sinusoidal tone generator with error envelopes."""
    return FlatTopGaussianEnvelope()


@pytest.fixture
def gen(tone):
    """Return a sinusoidal tone generator object."""
    gen = IQMixer(envelopes=[tone])
    return gen


@pytest.fixture
def hamiltonian(gen):
    """Return a resonator object."""

    def _method(dimension):
        drive = DriveOperator(gen, is_longitudinal=False)
        return Resonator(
            dimension=dimension,
            frequency=Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ),
            drives=[drive],
        )

    return _method


@pytest.fixture
def open_resonator():
    """Return an open model for the resonator."""
    tone = ZeroEnvelope()
    generator = IQMixer(envelopes=[tone])
    drive = DriveOperator(generator, is_longitudinal=False)
    resonator = Resonator(
        frequency=Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ),
        drives=[drive],
        dimension=DIMS,
    )
    resonator.t1 = T1
    resonator.temp = TEMP
    resonator.t2star = T2STAR
    model = OpenSystem(resonator)

    return model


@pytest.fixture
def expm(open_resonator):
    init = np.zeros((DIMS, 1), dtype=np.complex128)
    init[DIMS - 1][0] = 1  # Fully excited state
    init_dm = np.matmul(init, init.T)

    prop = ScipyExpm(open_resonator, res=100e9)
    prop.set_initial_state(init_dm)
    return prop


@pytest.fixture
def ode(open_resonator):
    init = np.zeros((DIMS, 1), dtype=np.complex128)
    init[DIMS - 1][0] = 1  # Fully excited state
    init_dm = np.matmul(init, init.T)

    open_resonator.ode_propagation = True
    prop = Vern7(open_resonator, res=100e9)
    prop.set_initial_state(init_dm)
    return prop


def test_get_matrix(hamiltonian, time_samples):
    """Test the getMatrix method."""
    for dim in np.arange(1, 10):
        hamil = hamiltonian(dim)
        hams = hamil.get_matrix(time_samples)
        assert hams.shape == time_samples.shape + (dim, dim)


def test_gradient(gen, hamiltonian, time_samples):
    """Test the gradients of the system.

    The Hamiltonian should have all derivatives of the drive
    plus the derivative w.r.t. the resonator frequency.

    """
    for dim in np.arange(1, 10):
        hamil = hamiltonian(dim)
        hamil.set_optimizable_parameters(hamil.get_parameters())
        grads = gen.generate_signal_gradient(time_samples)
        ham_grads = hamil.gradient(time_samples)
        assert ham_grads.shape == (grads.shape[0], grads.shape[1] + 1, dim, dim)


def test_decay_expm(expm):
    t_final = 20e-9
    ts = np.linspace(0, t_final, 101)

    states = expm.propagate(ts)
    final_state = states[-1]

    # Check if final state is density matrix
    assert np.isclose(np.linalg.trace(final_state), 1)

    # Check the excited state population
    assert np.isclose(final_state[DIMS - 1, DIMS - 1], 0)

    # Check the ground state population
    assert np.isclose(final_state[0, 0], 1)


def test_decay_ode(ode):
    t_final = 20e-9
    ts = np.linspace(0, t_final, 101)

    states = ode.propagate(ts)
    final_state = states[-1]

    # Check if final state is density matrix
    assert np.isclose(np.linalg.trace(final_state), 1)

    # Check the excited state population
    assert np.isclose(final_state[DIMS - 1, DIMS - 1], 0)

    # Check the ground state population
    assert np.isclose(final_state[0, 0], 1)
