"""Test composite optimization."""

import numpy as np
import pytest

from paraqeet.measurement.unitary_fidelity import UnitaryFidelity
from paraqeet.model.closed_system import ClosedSystem
from paraqeet.model.composite_hamiltonian import CompositeHamiltonian
from paraqeet.model.coupling import Coupling
from paraqeet.model.drive_operator import DriveOperator
from paraqeet.model.transmon import Transmon
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.scipy_optimizer import ScipyOptimizer
from paraqeet.optimizers.scipy_optimizer_gradient import ScipyOptimizerGradient
from paraqeet.propagation.scipy_expm_goat import ScipyExpmGOAT
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import FlatTopGaussianEnvelope
from paraqeet.signal.iq_mixer import IQMixer

FREQ1 = 5.5e9 * 2 * np.pi
ANHARM1 = -240e6 * 2 * np.pi

FREQ2 = 6.0e9 * 2 * np.pi
ANHARM2 = -200e6 * 2 * np.pi

COUPLINGSTR = 25e6 * 2 * np.pi

T_FINAL = 150e-9
TIMES = np.array([T_FINAL])
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
def coupled_transmons(tone):
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
    drive1 = DriveOperator(generator1, is_longitudinal=False)

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
    drive2 = DriveOperator(generator2, is_longitudinal=False)

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

    pauli_x = np.array([[0.0, 1], [1, 0.0]])
    pauli_z = np.array([[1, 0], [0.0, -1]])
    pauli_zx = np.exp(1j * np.pi / 4) * np.kron(pauli_z, pauli_x)
    cr_gate = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0], [0, 0, 1.0, 0]])

    cr_gate = pauli_zx @ cr_gate
    prop.set_initial_state(np.identity(9))
    gate_fid = UnitaryFidelity(
        propagation=prop,
        gate=cr_gate,
    )

    tone1_amp = tone1.get_parameters()[0]

    optmap = OptimizationMap()
    optmap.add(tone1, [tone1_amp])
    return gate_fid, optmap


@pytest.fixture
def opt(coupled_transmons):
    """Return Scipy optimizer from coupled transmons."""
    measure, optmap = coupled_transmons
    opt = ScipyOptimizer(measure, optimizables=optmap, times=TIMES)
    opt.set_options({"maxiter": 5})
    return opt


@pytest.fixture
def grad_opt(coupled_transmons):
    """Return Scipy optimizer gradient."""
    measure, optmap = coupled_transmons
    opt = ScipyOptimizerGradient(measure, optimizables=optmap, times=TIMES)
    opt.set_options({"maxiter": 2})
    return opt


def test_optim_finite_diff(opt):
    """Test optimization via finite differences."""
    res = opt.optimize()
    assert res.value < 0.1


def test_optim_goat(grad_opt):
    """Test GOAT optimization."""
    res = grad_opt.optimize()
    assert res.value < 0.1
