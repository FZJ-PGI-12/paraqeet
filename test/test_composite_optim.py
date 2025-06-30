"""Test composite optimisation."""

import pytest
import numpy as np

from paraQeet.model.drive_operator import DriveOperator
from paraQeet.optimisers.scipy_optimiser import ScipyOptimiser
from paraQeet.signal.envelopes import FlatTopGaussianEnvelope
from paraQeet.signal.iq_mixer import IQMixer

from paraQeet.optimisation_map import OptimisationMap
from paraQeet.quantity import Quantity
from paraQeet.measurement.unitary_fidelity import UnitaryFidelity
from paraQeet.model.coupling import Coupling
from paraQeet.optimisers.scipy_optimiser_gradient import ScipyOptimiserGradient
from paraQeet.propagation.scipy_expm_goat import ScipyExpmGOAT

from paraQeet.model.closed_system import ClosedSystem
from paraQeet.model.composite_hamiltonian import CompositeHamiltonian
from paraQeet.model.transmon import Transmon

FREQ1 = 5.5e9 * 2 * np.pi
ANHARM1 = -240e6 * 2 * np.pi

FREQ2 = 6.0e9 * 2 * np.pi
ANHARM2 = -200e6 * 2 * np.pi

COUPLINGSTR = 25e6 * 2 * np.pi

T_FINAL = 150e-9
RES = 100e9


@pytest.fixture
def tone():
    """Create a signal tone."""

    def _method(amp, t_final):
        tone = FlatTopGaussianEnvelope(
            amplitude=Quantity(
                amp * 2 * np.pi,
                min_value=0.8 * amp * 2 * np.pi,
                max_value=1.2 * amp * 2 * np.pi,
                unit="Hz",
            ),
            t_final=Quantity(
                t_final,
                min_value=0.8 * t_final,
                max_value=1.2 * t_final,
                unit="Hz",
            ),
        )
        return tone

    return _method


@pytest.fixture
def coupledTransmons(tone):
    """Create a coupled Transmon system."""
    freq1 = 6.0e9
    freq2 = 6.0002e9

    tone1 = tone(191e6, T_FINAL)
    tone2 = tone(9.18e6, T_FINAL)

    generator1 = IQMixer(
        envelopes=[tone1],
        frequency=Quantity(
            freq1 * 2 * np.pi,
            min_value=np.array(0.8 * freq1 * 2 * np.pi),
            max_value=np.array(1.2 * freq1 * 2 * np.pi),
            unit="Hz",
        ),
        phase=Quantity(
            0.01,
            min_value=np.array(-np.pi),
            max_value=np.array(np.pi),
            unit="rad",
        ),
    )
    drive1 = DriveOperator(generator1, isLongitudinal=False)

    generator2 = IQMixer(
        envelopes=[tone2],
        frequency=Quantity(
            freq2 * 2 * np.pi,
            min_value=np.array(0.8 * freq2 * 2 * np.pi),
            max_value=np.array(1.2 * freq2 * 2 * np.pi),
            unit="Hz",
        ),
        phase=Quantity(
            -0.39720756,
            min_value=np.array(-np.pi),
            max_value=np.array(np.pi),
            unit="rad",
        ),
    )
    drive2 = DriveOperator(generator2, isLongitudinal=False)

    transmon1 = Transmon(
        dimension=3,
        frequency=Quantity(FREQ1, np.array(0.8 * FREQ1), np.array(1.2 * FREQ1), "Hz"),
        anharmonicity=Quantity(ANHARM1, np.array(1.2 * ANHARM1), np.array(0.8 * ANHARM1), "Hz"),
        drives=[drive1],
    )
    transmon2 = Transmon(
        dimension=3,
        frequency=Quantity(FREQ2, np.array(0.8 * FREQ2), np.array(1.2 * FREQ2), "Hz"),
        anharmonicity=Quantity(ANHARM2, np.array(1.2 * ANHARM2), np.array(0.8 * ANHARM2), "Hz"),
        drives=[drive2],
    )
    coupling = Coupling(
        [transmon1, transmon2],
        is_longitudinal=False,
        coefficient=Quantity(
            COUPLINGSTR,
            np.array(0.8 * COUPLINGSTR),
            np.array(1.2 * COUPLINGSTR),
            "Hz",
        ),
    )
    hamiltonian = CompositeHamiltonian([transmon1, transmon2], [coupling])
    model = ClosedSystem(hamiltonian)
    prop = ScipyExpmGOAT(model=model, res=RES)

    X = np.array([[0.0, 1], [1, 0.0]])
    Z = np.array([[1, 0], [0.0, -1]])
    ZX = np.exp(1j * np.pi / 4) * np.kron(Z, X)
    CRGate = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0], [0, 0, 1.0, 0]])

    CRGate = ZX @ CRGate
    prop.set_initial_state(np.identity(9))
    gateFid = UnitaryFidelity(
        propagation=prop,
        gate=CRGate,
        times=np.array([0.0, T_FINAL]),
    )
    gateFid.restrict_subsystems([3, 3], [2, 2])

    tone1Amp = tone1.get_parameters()[0]

    optmap = OptimisationMap()
    optmap.add(tone1, [tone1Amp])
    return gateFid, optmap


@pytest.fixture
def opt(coupledTransmons):
    """Return Scipy optimiser from coupled transmons."""
    measure, optmap = coupledTransmons
    opt = ScipyOptimiser(measure, optimisables=optmap)
    opt.set_options({"maxiter": 5})
    return opt


@pytest.fixture
def gradOpt(coupledTransmons):
    """Return Scipy optimiser gradient."""
    measure, optmap = coupledTransmons
    opt = ScipyOptimiserGradient(measure, optimisables=optmap)
    opt.set_options({"maxiter": 2})
    return opt


def test_optim_finite_diff(opt):
    """Test optimisation via finite differences."""
    res = opt.optimise()
    assert res.value < 0.1


def test_optim_GOAT(gradOpt):
    """Test GOAT optimisation."""
    res = gradOpt.optimise()
    assert res.value < 0.1
