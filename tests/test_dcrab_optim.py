"""Test dCRAB optimization using GOAT over GRAPE. This is same as the example 02E_GOAToverGRAPE_dCRAB"""

import numpy as np
import pytest

from paraqeet.measurement.goat_over_grape import GOATOverGRAPE
from paraqeet.measurement.state_transfer_fidelity import StateTransferFidelityGRAPE
from paraqeet.model.rotating_frame import RotatingFrameDrive
from paraqeet.model.schroedinger_equation import SchroedingerEquation
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.dcrab_optimizer_gradient import DCRABOptimizerGradient
from paraqeet.propagation.scipy_expm_grape import ScipyExpmGRAPE
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import DCRABEnvelope
from paraqeet.signal.pwc_generator import PWCGenerator
from paraqeet.signal.waveform import FlatTopGaussianFilter
from tests.model.spin_rwa import SpinRWA

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
    spin = SpinRWA(drives=[drive])
    model = SchroedingerEquation(spin)
    return model


@pytest.fixture
def prop(model):
    prop = ScipyExpmGRAPE(model, resolution=1e9)

    init = np.array([[1.0], [0]])  # |0>
    target = np.array([[0.0], [1]])  # |1>

    prop.set_initial_state(init)
    prop.set_target_state(target)
    return prop


@pytest.fixture
def fid(prop):
    init = np.array([[1.0], [0]])  # |0>
    target = np.array([[0.0], [1]])  # |1>

    zeroone = StateTransferFidelityGRAPE(
        propagation=prop,
        initial_state=init,
        target_state=target,
    )
    return zeroone


@pytest.fixture
def opt_grad(tone, fid, gen, prop):
    optmap = OptimizationMap()
    params = tone.get_parameters()
    optmap.add(tone, [params[0]] + params[2:])
    optmap.register_params_with_optimizables()

    goat = GOATOverGRAPE(fid, prop, generators=[gen])
    opt_grad = DCRABOptimizerGradient(
        goat,
        optimization_map=optmap,
        super_iteration_every=150,
        max_super_iteration_num=5,
        print_every_iteration_num=10,
        super_iteration_tol=1e-9,
        seed=19573,
    )
    return opt_grad


def test_goat_over_grape(opt_grad):
    res = opt_grad.optimize(TLIST)
    assert res.value < 1e-4
