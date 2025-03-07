"""Test the composite Hamiltonian model."""

import pytest
import numpy as np

from cthree.quantity import Quantity
from cthree.model.coupling import Coupling
from cthree.signal.envelopes import FlatTopGaussianEnvelope
from cthree.signal.iq_mixer import IQMixer
from cthree.model.drive_operator import DriveOperator
from cthree.model.transmon import Transmon
from cthree.model.composite_hamiltonian import CompositeHamiltonian


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
    tone.set_optimisable_parameters(tone.get_parameters())
    return tone


@pytest.fixture
def gen(tone):
    """Return a sinusoidal generator object."""
    gen = IQMixer(envelopes=[tone])
    return gen


@pytest.fixture
def drive(gen):
    """Return a generator drive object."""
    drive = DriveOperator(gen, isLongitudinal=False)
    return drive


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
def transmon(transmon_parameters, drive):
    """Return a transmon generating function."""

    class CreateTransmon:
        def get(self, dimension):
            freq, anharm = transmon_parameters.get()
            transmon = Transmon(
                dimension=dimension,
                frequency=Quantity(freq, 0.8 * freq, 1.2 * freq),
                anharmonicity=Quantity(anharm, 1.2 * anharm, 0.8 * anharm),
                drives=[drive],
            )
            return transmon

    return CreateTransmon()


@pytest.fixture
def uncoupled_transmons(transmon):
    """Return a composite Hamiltonian generating function."""

    def _method(dim1, dim2):
        transmon1 = transmon.get(dim1)
        transmon2 = transmon.get(dim2)
        compositeHams = CompositeHamiltonian([transmon1, transmon2])
        return compositeHams

    return _method


@pytest.fixture
def coupled_transmons(transmon):
    """Return a coupled transmon generating function."""

    def _method(dim1: int, dim2: int, useRWA: bool = False):
        transmon1 = transmon.get(dim1)
        transmon2 = transmon.get(dim2)

        couplingStr = np.abs(transmon1.frequency.get_value() - transmon2.frequency.get_value()) * 0.05
        coupling = Coupling(
            [transmon1, transmon2],
            is_longitudinal=False,
            coefficient=Quantity(couplingStr, 0.8 * couplingStr, 1.2 * couplingStr, "Hz"),
            useRWA=useRWA,
        )

        compositeHams = CompositeHamiltonian([transmon1, transmon2], [coupling])
        return compositeHams

    return _method


def test_dimension(coupled_transmons):
    """Test dimension of matrix produced by compositeHamiltonian."""
    for _ in np.arange(1, 10):
        dim1 = np.random.randint(2, 6)
        dim2 = np.random.randint(2, 7)
        H = coupled_transmons(dim1, dim2)
        assert H.dimension() == dim1 * dim2


def test_get_matrix_one_time(uncoupled_transmons):
    """Test shape of Matrix produced by compositeHamiltonian."""
    for _ in np.arange(1, 10):
        dim1 = np.random.randint(2, 6)
        dim2 = np.random.randint(2, 7)
        H = uncoupled_transmons(dim1, dim2)
        hams = H.get_matrix_one_time(0)
        assert hams.shape == (dim1 * dim2, dim1 * dim2)


def test_get_matrix_one_time_rwa(coupled_transmons):
    """Test shape of Matrix produced by compositeHamiltonian."""
    for _ in np.arange(1, 10):
        dim1 = np.random.randint(2, 6)
        dim2 = np.random.randint(2, 7)
        H = coupled_transmons(dim1, dim2, useRWA=True)
        hams = H.get_matrix_one_time(0)
        assert hams.shape == (dim1 * dim2, dim1 * dim2)


def test_get_matrix(coupled_transmons, time_samples):
    """Test shape of Matrix produced by compositeHamiltonian."""
    for _ in np.arange(1, 10):
        dim1 = np.random.randint(2, 6)
        dim2 = np.random.randint(2, 7)
        H = coupled_transmons(dim1, dim2)
        hams = H.get_matrix(time_samples)
        assert hams.shape == time_samples.shape + (dim1 * dim2, dim1 * dim2)


def test_gradient(gen, coupled_transmons, time_samples):
    """Test shape of gradients by compositeHamiltonian.

    Number of gradient parameters include gradients from both the drives, and
    both the transmon frequency, anharmonicity and the coupling.
    """
    for _ in np.arange(1, 5):
        dim1 = np.random.randint(2, 6)
        dim2 = np.random.randint(2, 7)
        H = coupled_transmons(dim1, dim2)
        H.set_optimisable_parameters(H.get_parameters())
        grads = gen.generate_signal_gradient(time_samples)
        hamGrads = H.gradient(time_samples)
        assert hamGrads.shape == (
            grads.shape[0],
            grads.shape[1] * 2 + 5,
            dim1 * dim2,
            dim1 * dim2,
        )
