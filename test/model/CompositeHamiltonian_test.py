import pytest
import numpy as np

from cthree.Quantity import Quantity
from cthree.model.Coupling import Coupling
from cthree.signal.Device import CosToneErf
from cthree.signal.SimpleGenerator import CosGenerator
from cthree.model.GeneratorDrive import GeneratorDrive
from cthree.model.Transmon import Transmon
from cthree.model.CompositeHamiltonian import CompositeHamiltonian


LEN_SIG = 101


@pytest.fixture
def time_samples():
    return np.linspace(0, 10e-9, LEN_SIG)


@pytest.fixture
def tone():
    tone = CosToneErf()
    tone.setOptimisableParameters(tone.getParameters())
    return tone


@pytest.fixture
def gen(tone):
    gen = CosGenerator(devices=[tone])
    return gen


@pytest.fixture
def drive(gen):
    drive = GeneratorDrive(gen, isLongitudinal=False)
    return drive


@pytest.fixture
def transmonParameters():
    class RandomParameters:
        def get(self):
            freq = np.random.uniform(5.5, 6.0) * 1e9 * 2 * np.pi
            anharm = -np.random.uniform(2.0, 2.4) * 1e7 * 2 * np.pi
            return (freq, anharm)

    return RandomParameters()


@pytest.fixture
def transmon(transmonParameters, drive):
    class createTransmon:
        def get(self, dimension):
            freq, anharm = transmonParameters.get()
            transmon = Transmon(
                dimension=dimension,
                frequency=Quantity(freq, 0.8 * freq, 1.2 * freq),
                anharmonicity=Quantity(anharm, 1.2 * anharm, 0.8 * anharm),
                drives=[drive],
            )
            return transmon

    return createTransmon()


@pytest.fixture
def uncoupledTransmons(transmon):
    def _method(dim1, dim2):
        transmon1 = transmon.get(dim1)
        transmon2 = transmon.get(dim2)
        compositeHams = CompositeHamiltonian([transmon1, transmon2])
        return compositeHams

    return _method


@pytest.fixture
def coupledTransmons(transmon):
    def _method(dim1: int, dim2: int, useRWA: bool = False):
        transmon1 = transmon.get(dim1)
        transmon2 = transmon.get(dim2)

        couplingStr = (
            np.abs(
                transmon1.getFrequency().getValue()
                - transmon2.getFrequency().getValue()
            )
            * 0.05
        )
        coupling = Coupling(
            [transmon1, transmon2],
            isLongitudinal=False,
            coefficient=Quantity(
                couplingStr, 0.8 * couplingStr, 1.2 * couplingStr, "Hz"
            ),
            useRWA=useRWA,
        )

        compositeHams = CompositeHamiltonian([transmon1, transmon2], [coupling])
        return compositeHams

    return _method


def test_dimension(coupledTransmons):
    for _ in np.arange(1, 10):
        dim1 = np.random.randint(2, 6)
        dim2 = np.random.randint(2, 7)
        H = coupledTransmons(dim1, dim2)
        assert H.dimension() == dim1 * dim2


def test_getMatrixOneTime(uncoupledTransmons):
    """
    Test shape of Matrix produced by compositeHamiltonian.
    """
    for _ in np.arange(1, 10):
        dim1 = np.random.randint(2, 6)
        dim2 = np.random.randint(2, 7)
        H = uncoupledTransmons(dim1, dim2)
        hams = H.getMatrixOneTime(0)
        assert hams.shape == (dim1 * dim2, dim1 * dim2)


def test_getMatrixOneTime_RWA(coupledTransmons):
    """
    Test shape of Matrix produced by compositeHamiltonian.
    """
    for _ in np.arange(1, 10):
        dim1 = np.random.randint(2, 6)
        dim2 = np.random.randint(2, 7)
        H = coupledTransmons(dim1, dim2, useRWA=True)
        hams = H.getMatrixOneTime(0)
        assert hams.shape == (dim1 * dim2, dim1 * dim2)


def test_getMatrix(coupledTransmons, time_samples):
    """
    Test shape of Matrix produced by compositeHamiltonian.
    """
    for _ in np.arange(1, 10):
        dim1 = np.random.randint(2, 6)
        dim2 = np.random.randint(2, 7)
        H = coupledTransmons(dim1, dim2)
        hams = H.getMatrix(time_samples)
        assert hams.shape == time_samples.shape + (dim1 * dim2, dim1 * dim2)


def test_gradient(gen, coupledTransmons, time_samples):
    """
    Test shape of gradients by compositeHamiltonian.
    Number of gradient parameters include gradients from both the drives, and
    both the transmon frequency, anharmonicity and the coupling.
    """
    for _ in np.arange(1, 10):
        dim1 = np.random.randint(2, 6)
        dim2 = np.random.randint(2, 7)
        H = coupledTransmons(dim1, dim2)
        H.setOptimisableParameters(H.getParameters())
        grads = gen.generateSignalGradient(time_samples)
        hamGrads = H.gradient(time_samples)
        assert hamGrads.shape == (
            grads.shape[0],
            grads.shape[1] * 2 + 5,
            dim1 * dim2,
            dim1 * dim2,
        )
