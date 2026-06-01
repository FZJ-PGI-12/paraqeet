"""Test the Resonator model."""

import numpy as np
import pytest

from paraqeet.differentiable import Differentiable
from paraqeet.model.drive import Drive
from paraqeet.model.master_equation import MasterEquation
from paraqeet.model.resonator import Resonator
from paraqeet.propagation.scipy_expm import ScipyExpm
from paraqeet.propagation.utils import convert_dm_to_vec, lindblad_step
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

    def _method(num_fock):
        res = Resonator(
            num_fock=num_fock,
            frequency=Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ),
            drives=[],
        )
        drive_op = res.annihilation_op + (res.annihilation_op).conj().T
        drive = Drive(drive_op, gen)
        res.drives = [drive]
        return res

    return _method


@pytest.fixture
def open_resonator():
    """Return an open model for the resonator."""
    tone = ZeroEnvelope()
    generator = IQMixer(envelopes=[tone])
    res = Resonator(
        frequency=Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ),
        drives=[],
        num_fock=DIMS,
    )
    drive_op = res.annihilation_op + (res.annihilation_op).conj().T
    drive = Drive(drive_op, generator)
    res.drives = [drive]

    res.t1 = T1
    res.temp = TEMP
    res.t2star = T2STAR
    model = MasterEquation(
        hamiltonian_func=res.get_hamiltonian,
        hamiltonian_and_gradient_func=res.get_hamiltonian_and_gradient,
        jump_operators=res.get_jump_operators(),
    )

    return model


@pytest.fixture
def expm(open_resonator):
    init = np.zeros((DIMS, 1), dtype=np.complex128)
    init[DIMS - 1][0] = 1  # Fully excited state
    init_dm = np.matmul(init, init.T)
    init_dm_vec = convert_dm_to_vec(init_dm)

    prop = ScipyExpm(open_resonator.get_value, resolution=100e9, initial_state=init_dm_vec)
    return prop


@pytest.fixture
def ode(open_resonator):
    init = np.zeros((DIMS, 1), dtype=np.complex128)
    init[DIMS - 1][0] = 1  # Fully excited state
    init_dm = np.matmul(init, init.T)

    prop = Vern7(
        open_resonator.get_eom_ode_propagation,
        resolution=100e9,
        initial_state=init_dm,
        step_function=lindblad_step,
        jump_operators=open_resonator.jump_operators,
    )
    return prop


def test_get_hamiltonian(hamiltonian, time_samples):
    """Test the get_hamiltonian method."""
    for dim in np.arange(1, 10):
        hamil = hamiltonian(dim)
        hams = hamil.get_hamiltonian(time_samples)
        assert hams.shape == time_samples.shape + (dim, dim)


def test_gradient(gen, hamiltonian, time_samples):
    """Test the gradients of the system.

    The Hamiltonian should have all derivatives of the drive
    plus the derivative w.r.t. the resonator frequency.

    """
    for dim in np.arange(1, 10):
        hamil = hamiltonian(dim)
        if not isinstance(hamil, Differentiable):
            continue
        hamil.set_optimizable_parameters(hamil.get_parameters())
        _, grads = gen.get_value_and_gradient(time_samples)
        #  TODO: fix error related to the length of time_samples-array
        _, ham_grads = hamil.get_value_and_gradient(time_samples)
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
