import pytest
import numpy as np

from cthree.model.GeneratorDrive import GeneratorDrive
from cthree.optimisers.ScipyOptimiser import ScipyOptimiser
from cthree.signal.Device import CosToneErf
from cthree.signal.SimpleGenerator import CosGenerator

from cthree.OptimisationMap import OptimisationMap
from cthree.Quantity import Quantity
from cthree.measurement.UnitaryFidelity import UnitaryFidelity
from cthree.model.Coupling import Coupling
from cthree.optimisers.ScipyOptimiserGradient import ScipyOptimiserGradient
from cthree.propagation.ScipyExpmGOAT import ScipyExpmGOAT

from cthree.model.ClosedModel import ClosedModel
from cthree.model.CompositeHamiltonian import CompositeHamiltonian
from cthree.model.Transmon import Transmon

FREQ1 = 5.5e9 * 2 * np.pi
ANHARM1 = -240e6 * 2 * np.pi

FREQ2 = 6.0e9 * 2 * np.pi
ANHARM2 = -200e6 * 2 * np.pi

COUPLINGSTR = 25e6 * 2 * np.pi

T_FINAL = 150e-9
RES = 100e9


@pytest.fixture
def tone():
    def _method(amp, freq, phase, t_final):
        tone = CosToneErf()
        tone.amplitude = Quantity(
            amp * 2 * np.pi,
            min_value=0.8 * amp * 2 * np.pi,
            max_value=1.2 * amp * 2 * np.pi,
            unit="Hz",
        )
        tone.frequency = Quantity(
            freq * 2 * np.pi,
            min_value=0.8 * freq * 2 * np.pi,
            max_value=1.2 * freq * 2 * np.pi,
            unit="Hz",
        )
        tone.phase = Quantity(phase, min_value=-np.pi, max_value=np.pi, unit="Hz")
        tone.t_final = Quantity(
            t_final, min_value=0.8 * t_final, max_value=1.2 * t_final, unit="Hz"
        )
        return tone

    return _method


@pytest.fixture
def coupledTransmons(tone):
    tone1 = tone(1.91e8, 6.0e9, 0.01, T_FINAL)
    tone2 = tone(9.18e6, 6.0002e9, -0.39720756, T_FINAL)

    generator1 = CosGenerator(devices=[tone1])
    drive1 = GeneratorDrive(generator1, isLongitudinal=False)

    generator2 = CosGenerator(devices=[tone2])
    drive2 = GeneratorDrive(generator2, isLongitudinal=False)

    transmon1 = Transmon(
        dimension=3,
        frequency=Quantity(FREQ1, 0.8 * FREQ1, 1.2 * FREQ1, "Hz"),
        anharmonicity=Quantity(ANHARM1, 1.2 * ANHARM1, 0.8 * ANHARM1, "Hz"),
        drives=[drive1],
    )
    transmon2 = Transmon(
        dimension=3,
        frequency=Quantity(FREQ2, 0.8 * FREQ2, 1.2 * FREQ2, "Hz"),
        anharmonicity=Quantity(ANHARM2, 1.2 * ANHARM2, 0.8 * ANHARM2, "Hz"),
        drives=[drive2],
    )
    coupling = Coupling(
        [transmon1, transmon2],
        isLongitudinal=False,
        coefficient=Quantity(COUPLINGSTR, 0.8 * COUPLINGSTR, 1.2 * COUPLINGSTR, "Hz"),
    )
    hamiltonian = CompositeHamiltonian([transmon1, transmon2], [coupling])
    model = ClosedModel(hamiltonian)
    prop = ScipyExpmGOAT(model=model, res=RES)

    X = np.array([[0.0, 1], [1, 0.0]])
    Z = np.array([[1, 0], [0.0, -1]])
    ZX = np.exp(1j * np.pi / 4) * np.kron(Z, X)
    CRGate = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0], [0, 0, 1.0, 0]])

    CRGate = ZX @ CRGate
    prop.setInitialState(np.identity(9))
    gateFid = UnitaryFidelity(
        propagation=prop,
        gate=CRGate,
        times=np.array([0.0, T_FINAL]),
    )
    gateFid.restrictSubsystems([3, 3], [2, 2])

    optmap = OptimisationMap()
    optmap.add(tone1)
    return gateFid, optmap


@pytest.fixture
def opt(coupledTransmons):
    measure, optmap = coupledTransmons
    opt = ScipyOptimiser(measure, optimisables=optmap)
    opt.setOptions({"ftol": 0.1})
    return opt


@pytest.fixture
def gradOpt(coupledTransmons):
    measure, optmap = coupledTransmons
    opt = ScipyOptimiserGradient(measure, optimisables=optmap)
    opt.setOptions({"ftol": 0.1})
    return opt


def test_optim_finite_diff(opt):
    res = opt.optimise()
    assert res.value < 0.1


def test_optim_GOAT(gradOpt):
    res = gradOpt.optimise()
    assert res.value < 0.1
