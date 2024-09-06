import pytest
import numpy as np

from cthree.Quantity import Quantity
from cthree.model.Coupling import Coupling
from cthree.model.Transmon import Transmon

COUPLINGSTR = 25e6 * 2 * np.pi
LEN_SIG = 101


@pytest.fixture
def time_samples():
    return np.linspace(0, 10e-9, LEN_SIG)


@pytest.fixture
def transmonParameters():
    class RandomParameters:
        def get(self):
            freq = np.random.uniform(5.5, 6.0) * 1e9 * 2 * np.pi
            anharm = -np.random.uniform(2.0, 2.4) * 1e7 * 2 * np.pi
            return (freq, anharm)

    return RandomParameters()


@pytest.fixture
def transmon(transmonParameters):
    class createTransmon:
        def get(self, dimension):
            freq, anharm = transmonParameters.get()
            transmon = Transmon(
                dimension=dimension,
                frequency=Quantity(freq, 0.8 * freq, 1.2 * freq),
                anharmonicity=Quantity(anharm, 1.2 * anharm, 0.8 * anharm),
            )
            return transmon

    return createTransmon()


@pytest.fixture
def coupling(transmon):
    def _method(dim1: int, dim2: int, isLongitudinal: bool, useRWA: bool = False):
        transmon1 = transmon.get(dim1)
        transmon2 = transmon.get(dim2)
        coupling = Coupling(
            [transmon1, transmon2],
            isLongitudinal=isLongitudinal,
            useRWA=useRWA,
            coefficient=Quantity(
                COUPLINGSTR, 0.8 * COUPLINGSTR, 1.2 * COUPLINGSTR, "Hz"
            ),
        )

        return coupling

    return _method


def test_getMatricesOneTime(coupling):
    """
    Test shape of Matrix produced by the coupling Hamiltonian.
    """
    for _ in range(10):
        dim1 = np.random.randint(2, 7)
        dim2 = np.random.randint(2, 7)
        dims = [dim1, dim2]

        # Test shape for Longitudinal coupling
        coup = coupling(dim1, dim2, isLongitudinal=True)
        coup_hams = coup.getMatricesOneTime(0)
        for i, ops in enumerate(coup_hams):
            assert np.shape(ops) == (dims[i], dims[i])

        # Test Longitudinal couplings are diagonal
        for i, ops in enumerate(coup_hams):
            numNonZero = np.count_nonzero(ops - np.diag(np.diagonal(ops)))
            assert numNonZero == 0

        # Test for RWA
        coup = coupling(dim1, dim2, isLongitudinal=False, useRWA=True)
        coup_hams = coup.getMatricesOneTime(0)
        for term in coup_hams:
            for i, ops in enumerate(term):
                assert np.shape(ops) == (dims[i], dims[i])

        # Test shape for Transverse coupling
        coup = coupling(dim1, dim2, isLongitudinal=False)
        coup_hams = coup.getMatricesOneTime(0)
        for i, ops in enumerate(coup_hams):
            assert np.shape(ops) == (dims[i], dims[i])


def test_getMatrices(coupling, time_samples):
    for _ in range(10):
        dim1 = np.random.randint(2, 7)
        dim2 = np.random.randint(2, 7)
        dims = [dim1, dim2]

        # Test shape for Longitudinal coupling
        coup = coupling(dim1, dim2, isLongitudinal=True)
        coup_hams = coup.getMatrices(time_samples)
        for i, ops in enumerate(coup_hams):
            assert np.shape(ops) == time_samples.shape + (dims[i], dims[i])

        # Test for RWA
        coup = coupling(dim1, dim2, isLongitudinal=False, useRWA=True)
        coup_hams = coup.getMatrices(time_samples)
        for term in coup_hams:
            for i, ops in enumerate(term):
                assert np.shape(ops) == time_samples.shape + (dims[i], dims[i])

        # Test shape for Transverse coupling
        coup = coupling(dim1, dim2, isLongitudinal=False)
        coup_hams = coup.getMatrices(time_samples)
        for i, ops in enumerate(coup_hams):
            assert np.shape(ops) == time_samples.shape + (dims[i], dims[i])


def test_gradient_shape(coupling, time_samples):
    # Test if the coupling is not optimized
    dim1 = np.random.randint(2, 7)
    dim2 = np.random.randint(2, 7)
    coup = coupling(dim1, dim2, isLongitudinal=False)
    grads = coup.gradient(time_samples)
    assert grads.shape == (0, dim1 * dim2, dim1 * dim2)

    # Test with couping optimized
    dim1 = np.random.randint(2, 7)
    dim2 = np.random.randint(2, 7)
    dims = [dim1, dim2]
    coup = coupling(dim1, dim2, isLongitudinal=False)
    coup.setOptimisableParameters(coup.getParameters())
    grads = coup.gradient(time_samples)
    for grad in grads:
        for i, ops in enumerate(grad):
            assert np.shape(ops) == (dims[i], dims[i])
