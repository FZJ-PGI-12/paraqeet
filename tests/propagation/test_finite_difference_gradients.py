"""Test the finite difference gradients against the analytic gradient methods."""

import numpy as np
import pytest

from paraqeet.eom.schroedinger_equation import SchroedingerEquation
from paraqeet.exceptions import ConfigurationException
from paraqeet.hamiltonian.drive import Drive
from paraqeet.hamiltonian.qubit import QubitHamiltonian
from paraqeet.optimization_map import OptimizationMap
from paraqeet.propagation.auto_diff_gradients import AutoDiffGradients
from paraqeet.propagation.expm import Expm
from paraqeet.propagation.finite_difference_gradients import FiniteDifferenceGradients
from paraqeet.propagation.goat import GOAT
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import GaussEnvelope

T_FINAL = 20e-9
FREQ = 1e6
TLIST = np.linspace(0, T_FINAL, 6)
RESOLUTION = 10e9

INIT_STATE = np.array([[1.0], [0.0]], dtype=np.complex128)


@pytest.fixture
def drive_amplitudes():
    """Return the two drive amplitudes, the free parameters of the test model."""
    return [
        Quantity(np.pi / T_FINAL / 3, -np.pi / T_FINAL, np.pi / T_FINAL),
        Quantity(np.pi / T_FINAL / 5, -np.pi / T_FINAL, np.pi / T_FINAL),
    ]


@pytest.fixture
def optimization_map(drive_amplitudes):
    """Return the optimization map of a qubit driven by two Gaussian tones."""
    optmap = OptimizationMap()
    for amplitude in drive_amplitudes:
        tone = GaussEnvelope(amplitude=amplitude)
        tone.t_final.set_value(T_FINAL)
        optmap.add(tone, [amplitude])
    optmap.register_params_with_optimizables()
    return optmap


@pytest.fixture
def schroedinger(optimization_map):
    """Return the Schrödinger equation of the driven qubit."""
    hamiltonian = QubitHamiltonian(Quantity(FREQ, FREQ / 4, FREQ), drives=[])
    hamiltonian.drives = [
        Drive(hamiltonian.sigma_minus, tone, add_hermitian=True) for tone in optimization_map.get_optimizables()
    ]
    return SchroedingerEquation(
        hamiltonian_func=hamiltonian.get_value,
        hamiltonian_gradient_func=hamiltonian.get_gradient,
    )


def test_finite_differences_match_automatic_differentiation(schroedinger, optimization_map):
    """The central difference of the propagation reproduces the gradient of every parameter."""
    propagation = Expm(eom_func=schroedinger.get_value, resolution=RESOLUTION, initial_state=INIT_STATE)
    finite_difference = FiniteDifferenceGradients(propagation, optimization_map)
    autodiff = AutoDiffGradients(propagation, eom_gradient_func=schroedinger.get_gradient)

    expected = np.array(autodiff.get_gradient(TLIST))
    gradient = np.array(finite_difference.get_gradient(TLIST))

    assert gradient.shape == expected.shape == (len(TLIST), 2) + INIT_STATE.shape
    assert np.abs(expected).max() > 1e-12
    np.testing.assert_allclose(gradient, expected, rtol=1e-5, atol=1e-5 * np.abs(expected).max())


def test_finite_differences_match_goat(schroedinger, optimization_map):
    """The same holds for the GOAT gradient, which solves the extended equation of motion."""
    propagation = Expm(eom_func=schroedinger.get_value, resolution=RESOLUTION, initial_state=INIT_STATE)
    finite_difference = FiniteDifferenceGradients(propagation, optimization_map)
    goat = GOAT(propagation, eom_gradient_func=schroedinger.get_gradient)

    value, gradient = finite_difference.get_value_and_gradient(TLIST)
    expected_value, expected_gradient = goat.get_value_and_gradient(TLIST)

    np.testing.assert_allclose(value, np.array(expected_value), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        gradient, np.array(expected_gradient), rtol=1e-5, atol=1e-5 * np.abs(np.array(expected_gradient)).max()
    )


def test_finite_differences_restore_the_parameters(schroedinger, optimization_map, drive_amplitudes):
    """The displaced parameters are put back, also when the propagation raises."""
    propagation = Expm(eom_func=schroedinger.get_value, resolution=RESOLUTION, initial_state=INIT_STATE)
    finite_difference = FiniteDifferenceGradients(propagation, optimization_map)
    values = [np.array(amplitude.get_value()) for amplitude in drive_amplitudes]

    finite_difference.get_gradient(TLIST)
    for amplitude, value in zip(drive_amplitudes, values, strict=True):
        np.testing.assert_array_equal(np.array(amplitude.get_value()), value)

    with pytest.raises(ValueError):
        finite_difference.get_gradient(np.array([0.0]))
    for amplitude, value in zip(drive_amplitudes, values, strict=True):
        np.testing.assert_array_equal(np.array(amplitude.get_value()), value)


def test_finite_differences_reject_an_invalid_configuration(schroedinger, optimization_map):
    """The displacement has to be positive, and there have to be parameters to displace."""
    propagation = Expm(eom_func=schroedinger.get_value, resolution=RESOLUTION, initial_state=INIT_STATE)

    with pytest.raises(ConfigurationException):
        FiniteDifferenceGradients(propagation, optimization_map, epsilon=0.0)

    finite_difference = FiniteDifferenceGradients(propagation, optimization_map)
    with pytest.raises(ConfigurationException):
        finite_difference.epsilon = -1e-6

    with pytest.raises(ConfigurationException):
        FiniteDifferenceGradients(propagation, OptimizationMap()).get_gradient(TLIST)

    with pytest.raises(ConfigurationException):
        finite_difference._eom_and_gradient_func(TLIST)
