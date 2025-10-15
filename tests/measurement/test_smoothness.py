import numpy as np
import pytest

from paraqeet.measurement.smoothness import Smoothness
from paraqeet.optimization_map import OptimizationMap
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import GaussEnvelope
from paraqeet.signal.pwc_generator import PWCGenerator


@pytest.fixture
def pwc_gen():
    delta_sampling = 33e-9
    n_pwc = 40  # number of piecewise constants in the pulse
    t_final = n_pwc * delta_sampling
    tlist = np.linspace(0, t_final, n_pwc + 1)
    eps_qubit = 2 * np.pi * 1.0  # (in MHz)
    eps_max_qubit = 5 * eps_qubit  # (in MHz)
    tone = GaussEnvelope(
        amplitude=Quantity(eps_qubit * 1e6, -eps_max_qubit * 1e6, eps_max_qubit * 1e6),
        t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final),
    )
    return PWCGenerator(envelopes=[tone], max_amplitude=eps_max_qubit * 1e6, tlist=tlist)


@pytest.fixture
def another_tone():
    t_final = 10e-9
    eps = 2 * np.pi * 0.7  # (in MHz)
    eps_max = 5 * eps  # (in MHz)
    tone = GaussEnvelope(
        amplitude=Quantity(eps * 1e6, -eps_max * 1e6, eps_max * 1e6),
        t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final),
    )
    return tone


def test_smoothness_init(pwc_gen):
    smoothness = Smoothness(pwc_generator=pwc_gen)
    assert smoothness is not None


def test_smoothness_measure(pwc_gen):
    smoothness = Smoothness(pwc_generator=pwc_gen)
    expected_measured_value = 0.9999547789963114
    assert np.abs(smoothness.calculate_normalized_scalar() - expected_measured_value) < 1e-8


def test_smoothness_gradient(pwc_gen, another_tone):
    smoothness = Smoothness(pwc_generator=pwc_gen)
    optmap = OptimizationMap()
    optmap.add(pwc_gen, pwc_gen.get_parameters())
    # We add dummy parameters to check if the gradient is computed
    # correctly by padding zeros
    # optmap.add(tone_qubit, tone_qubit.get_parameters())
    optmap.add(another_tone, another_tone.get_parameters())
    optmap.register_params_with_optimizables()
    assert len(optmap.get_all_parameters()) == len(another_tone.optimizable_parameters) + len(
        pwc_gen.optimizable_parameters
    )
    _, gradient = smoothness.calculate_normalized_scalar_and_gradient()
    num_opt_params = 0
    for param in optmap.get_all_parameters():
        num_opt_params += param.get_value().shape[0]
    assert gradient.shape[0] == num_opt_params
    assert np.abs(gradient[6] - 5.39894375e-13) / 5.39894375e-13 < 1e-5  # precomputed value
    assert np.abs(gradient[39] - 4.626191950335811e-14) / 4.626191950335811e-14 < 1e-5  # precomputed value
    # since the gradient of smoothness does not depend
    # on the parameters of tone we can also check:
    assert gradient[-1] == 0.0
    assert gradient[-2] == 0.0
