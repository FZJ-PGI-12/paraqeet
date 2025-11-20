"""Test dCRAB optimization using GOAT over GRAPE. This is same as the example 02E_GOAToverGRAPE_dCRAB"""

import pytest

import numpy as np
from paraqeet.quantity import Quantity
from tests.model.tls import TLS

from paraqeet.signal.pwc_generator import PWCGenerator
from paraqeet.signal.envelopes import DCRABEnvelope
from paraqeet.signal.waveform import FlatTopGaussianFilter
from paraqeet.model.closed_system import ClosedSystem
from paraqeet.model.rotating_frame_drive import RotatingFrameDrive
from paraqeet.measurement.state_transfer_fidelity import StateTransferFidelityGRAPE
from paraqeet.propagation.scipy_expm_grape import ScipyExpmGRAPE
from paraqeet.optimisation_map import OptimisationMap
from paraqeet.optimisers.dcrab_optimiser_gradient import DCRABOptimiserGradient
from paraqeet.measurement.goat_over_grape import GOATOverGRAPE

T_FINAL = 20e-9
TLIST = np.linspace(0, T_FINAL, 40)
EPS = 5 * np.pi * 1.0  # initial amplitude of the resonator (in MHz)
EPS_MAX = 10 * EPS  # maximum amplitude of the resonator (in MHz)


@pytest.fixture
def tone():
    tone = DCRABEnvelope(
        num_components=2,
        max_frequency=5 * 2 * np.pi,
        amplitude=Quantity(EPS * 1e6, -EPS_MAX * 1e6, EPS_MAX * 1e6, name="Amplitude"),
        t_final=Quantity(T_FINAL, 1 / 2 * T_FINAL, 2 * T_FINAL, name="t_final"),
        seed=18537,
    )
    return tone


@pytest.fixture
def gen(tone):
    smooth_tone = FlatTopGaussianFilter(
        envelopes=tone, t_final=Quantity(T_FINAL, 1 / 2 * T_FINAL, 2 * T_FINAL, name="t_final")
    )
    gen = PWCGenerator(envelopes=[smooth_tone], tlist=TLIST, max_amplitude=np.sqrt(2) * EPS_MAX * 1e6)
    return gen


@pytest.fixture
def model(gen):
    drive = RotatingFrameDrive(gen)
    spin = TLS(drives=[drive])
    model = ClosedSystem(spin)
    return model


@pytest.fixture
def fid(model):
    prop = ScipyExpmGRAPE(model, res=1e9)

    init = np.array([[1.0], [0]])  # |0>
    target = np.array([[0.0], [1]])  # |1>

    prop.set_initial_state(init)
    prop.target_state = target

    zeroone = StateTransferFidelityGRAPE(
        propagation=prop,
        initial_state=init,
        target_state=target,
        times=TLIST,
    )
    return zeroone


@pytest.fixture
def opt_grad(tone, fid, gen):
    optmap = OptimisationMap()
    params = tone.get_parameters()
    optmap.add(tone, [params[0]] + params[2:])
    optmap.register_params_with_optimisables()

    goat = GOATOverGRAPE(fid, generators=[gen], generators_order=[0])
    opt_grad = DCRABOptimiserGradient(
        goat,
        optimisation_map=optmap,
        super_iteration_every=150,
        max_super_iteration_num=5,
        print_every_iteration_num=10,
        super_iteration_tol=1e-9,
        seed=19573,
    )
    return opt_grad


def test_goat_over_grape(opt_grad):
    res = opt_grad.optimise()
    assert res.value < 1e-4
