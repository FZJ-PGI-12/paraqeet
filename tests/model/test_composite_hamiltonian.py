"""Test the composite Hamiltonian model."""

import numpy as np
import pytest

from paraqeet.model.composite_system import CompositeSystem
from paraqeet.model.coupling import TwoBodyCoupling
from paraqeet.model.drive import Drive
from paraqeet.model.transmon import Transmon
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import FlatTopGaussianEnvelope
from paraqeet.signal.iq_mixer import IQMixer

LEN_SIG = 101


@pytest.fixture
def time_samples():
    """Generate time samples according to the given signal length.

    The given signal length `LEN_SIG` is a module level global variable.

    """
    return np.linspace(0, 10e-9, LEN_SIG)


@pytest.fixture
def tone():
    """Return a cosine tone."""
    tone = FlatTopGaussianEnvelope()
    tone.set_optimizable_parameters(tone.get_parameters())
    return tone


@pytest.fixture
def gen(tone):
    """Return a sinusoidal generator object."""
    gen = IQMixer(envelopes=[tone])
    return gen


@pytest.fixture
def transmon_parameters():
    """Return a random parameter generating function for transmons."""

    class RandomParameters:
        def get(self):
            freq = np.random.uniform(5.5, 6.0) * 1e9 * 2 * np.pi
            anharm = -np.random.uniform(2.0, 2.4) * 1e7 * 2 * np.pi
            return (freq, anharm)

    return RandomParameters()


@pytest.fixture
def transmon(transmon_parameters, gen):
    """Return a transmon generating function."""

    def get(num_levels):
        freq, anharm = transmon_parameters.get()
        transmon = Transmon(
            num_levels=num_levels,
            frequency=Quantity(freq, 0.8 * freq, 1.2 * freq),
            anharmonicity=Quantity(anharm, 1.2 * anharm, 0.8 * anharm),
            drives=[],
        )
        drive_op = transmon.annihilation_op + (transmon.annihilation_op).conj().T
        drive = Drive(drive_op, gen)
        transmon.drives = [drive]
        return transmon

    return get


@pytest.fixture
def uncoupled_transmons(transmon):
    """Return a CompositeSystem generating function."""

    def _method(dim1, dim2):
        transmon1 = transmon(dim1)
        transmon2 = transmon(dim2)
        composite_sys = CompositeSystem([transmon1, transmon2])
        return composite_sys

    return _method


@pytest.fixture
def coupled_transmons(transmon):
    """Return a coupled transmon generating function."""

    def _method(dim1: int, dim2: int, use_rwa: bool = False):
        transmon1 = transmon(dim1)
        transmon2 = transmon(dim2)

        coupling_str = np.abs(transmon1.frequency.get_value() - transmon2.frequency.get_value()) * 0.05
        coupling = TwoBodyCoupling(
            transmon1,
            transmon2,
            is_longitudinal=False,
            coefficient=Quantity(coupling_str, 0.8 * coupling_str, 1.2 * coupling_str, "Hz"),
            use_rwa=use_rwa,
        )

        composite_sys = CompositeSystem([transmon1, transmon2], [coupling])
        return composite_sys

    return _method


@pytest.fixture
def coupled_transmons_chain(transmon, random_quantity):
    def _method(dims: list[int], open_system: bool = True) -> CompositeSystem:
        transmons = [transmon(d) for d in dims]
        if open_system:
            for t in transmons:
                t.temp = random_quantity(1, "K")
                t.t1 = random_quantity(1, "")
                t.t2star = random_quantity(1, "")
        coupling_strengths = [
            0.5 * np.abs(transmons[i].frequency.get_value() - transmons[i + 1].frequency.get_value())
            for i in range(len(transmons) - 1)
        ]
        couplings = [
            TwoBodyCoupling(
                subsystem_A=transmons[i],
                subsystem_B=transmons[i + 1],
                is_longitudinal=False,
                coefficient=Quantity(
                    coupling_strengths[i], 0.8 * coupling_strengths[i], 1.2 * coupling_strengths[i], "Hz"
                ),
            )
            for i in range(len(transmons) - 1)
        ]

        return CompositeSystem(transmons, couplings)

    return _method


def test_dimension(coupled_transmons):
    """Test dimension of matrix produced by CompositeSystem."""
    for _ in np.arange(1, 10):
        dim1 = np.random.randint(2, 6)
        dim2 = np.random.randint(2, 7)
        hamil = coupled_transmons(dim1, dim2)
        assert hamil.dimension() == dim1 * dim2


def test_hamiltonian_at_timestep(uncoupled_transmons):
    """Test shape of matrix produced by CompositeSystem at each timestep."""
    for _ in np.arange(1, 10):
        dim1 = np.random.randint(2, 6)
        dim2 = np.random.randint(2, 7)
        hamil = uncoupled_transmons(dim1, dim2)
        hams = hamil.get_hamiltonian_at_timestep(0)
        assert hams.shape == (dim1 * dim2, dim1 * dim2)


def test_hamiltonian_at_timestep_rwa(coupled_transmons):
    """Test shape of matrix produced by CompositeSystem at each timestep under RWA."""
    for _ in np.arange(1, 10):
        dim1 = np.random.randint(2, 6)
        dim2 = np.random.randint(2, 7)
        hamil = coupled_transmons(dim1, dim2, use_rwa=True)
        hams = hamil.get_hamiltonian_at_timestep(0)
        assert hams.shape == (dim1 * dim2, dim1 * dim2)


def test_get_hamiltonian(coupled_transmons, time_samples):
    """Test shape of matrix produced by CompositeSystem for an array of timesteps."""
    for _ in np.arange(1, 10):
        dim1 = np.random.randint(2, 6)
        dim2 = np.random.randint(2, 7)
        hamil = coupled_transmons(dim1, dim2)
        hams = hamil.get_hamiltonian(time_samples)
        assert hams.shape == time_samples.shape + (dim1 * dim2, dim1 * dim2)


def test_gradient(gen, coupled_transmons, time_samples):
    """Test shape of gradients by CompositeSystem.

    Number of gradient parameters include gradients from both the drives, and
    both the transmon frequency, anharmonicity and the coupling.
    """
    for _ in np.arange(1, 5):
        dim1 = np.random.randint(2, 6)
        dim2 = np.random.randint(2, 7)
        hamil = coupled_transmons(dim1, dim2)
        hamil.set_optimizable_parameters(hamil.get_parameters())
        _, grads = gen.get_value_and_gradient(time_samples)
        _, ham_grads = hamil.get_hamiltonian_and_gradient(time_samples)
        assert ham_grads.shape == (
            grads.shape[0],
            grads.shape[1] * 2 + 5,
            dim1 * dim2,
            dim1 * dim2,
        )
