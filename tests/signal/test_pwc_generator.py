"""Test how a piecewise constant pulse is sampled, by the generator and by the propagation methods."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from paraqeet.eom.schroedinger_equation import SchroedingerEquation
from paraqeet.hamiltonian.drive import Drive
from paraqeet.hamiltonian.qubit import QubitHamiltonian
from paraqeet.optimization_map import OptimizationMap
from paraqeet.propagation.diffrax_ode import DiffraxODE
from paraqeet.propagation.expm import Expm
from paraqeet.propagation.utils import construct_batched_times, schrodinger_step
from paraqeet.propagation.vern7 import Vern7
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import GaussEnvelope
from paraqeet.signal.pwc_generator import PWCGenerator
from tests.propagation.test_common_propagation import make_propagation

T_FINAL = 20e-9
FREQ = 1e6
N_PIXELS = 20
INITIAL_PROPAGATOR = np.eye(2, dtype=np.complex128)


def make_generator(t_start=0.0, multiply_flat_top=False):
    """Return a PWC generator of a Gaussian, discretized on a grid starting at ``t_start``."""
    amplitude = np.pi / T_FINAL / 3
    tone = GaussEnvelope(amplitude=Quantity(amplitude, -4 * amplitude, 4 * amplitude))
    tone.t_final.set_value(T_FINAL)
    generator = PWCGenerator(envelopes=[tone], tlist=np.linspace(t_start, t_start + T_FINAL, N_PIXELS + 1))
    generator.multiply_flat_top = multiply_flat_top
    generator.max_amplitude = 2e8
    return generator


@pytest.fixture
def pwc_qubit():
    """Return the pulse-pixel grid, the Schroedinger equation and the generator of a driven qubit."""
    generator = make_generator(multiply_flat_top=True)

    hamiltonian = QubitHamiltonian(Quantity(FREQ, FREQ / 4, FREQ), drives=[])
    hamiltonian.drives = [Drive(hamiltonian.sigma_minus, generator, add_hermitian=True)]
    optmap = OptimizationMap()
    optmap.add(generator)
    optmap.register_params_with_optimizables()

    schroedinger = SchroedingerEquation(
        hamiltonian_func=hamiltonian.get_value,
        hamiltonian_gradient_func=hamiltonian.get_gradient,
    )
    return np.asarray(generator.tlist), schroedinger, generator


# --- Which pulse pixel a time belongs to ------------------------------------------------------


@pytest.mark.parametrize("t_start", [0.0, 5e-9])
def test_pixel_index_of_the_pixel_centers(t_start):
    """The center of a pixel belongs to that pixel, wherever the pulse starts.

    Dividing the time by the width of a pixel instead assumes a grid starting at zero, and shifts
    every index of a pulse that does not.
    """
    generator = make_generator(t_start)

    index = np.asarray(generator._compute_pixel_index(generator._time_grid))

    np.testing.assert_array_equal(index, np.arange(N_PIXELS))


@pytest.mark.parametrize("t_start", [0.0, 5e-9])
def test_value_at_a_pixel_center_is_that_pixels_amplitude(t_start):
    """The signal at the center of a pixel is the amplitude stored for that pixel."""
    generator = make_generator(t_start)

    value = np.asarray(generator.get_value(generator._time_grid))

    expected = np.asarray(generator._inphase.get_value()) + 1j * np.asarray(generator._outofphase.get_value())
    np.testing.assert_allclose(value, expected, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("t_start", [0.0, 5e-9])
def test_a_boundary_belongs_to_the_pixel_that_starts_there(t_start):
    """A pixel covers the half open interval ``[tlist[k], tlist[k + 1])``.

    A time on a boundary therefore agrees with one just after it, not one just before.
    """
    generator = make_generator(t_start)
    # The interior boundaries, which have a pulse pixel on either side.
    edges = np.asarray(generator.tlist)[1:-1]
    width = np.asarray(generator.tlist)[1] - np.asarray(generator.tlist)[0]
    centers = np.asarray(generator._time_grid)

    on_the_boundary = np.asarray(generator.get_value(edges))
    just_after = np.asarray(generator.get_value(edges + 1e-3 * width))
    just_before = np.asarray(generator.get_value(edges - 1e-3 * width))

    # Boundary k is the start of pixel k, so it carries the pixel that follows it.
    np.testing.assert_array_equal(on_the_boundary, just_after)
    np.testing.assert_array_equal(on_the_boundary, np.asarray(generator.get_value(centers[1:])))
    np.testing.assert_array_equal(just_before, np.asarray(generator.get_value(centers[:-1])))


def test_times_outside_the_pulse_are_clipped():
    """Times before and after the pulse fall back to the first and the last pixel."""
    generator = make_generator()

    index = np.asarray(generator._compute_pixel_index(np.array([-1e-9, 0.0, T_FINAL, T_FINAL + 1e-9])))

    np.testing.assert_array_equal(index, [0, 0, N_PIXELS - 1, N_PIXELS - 1])


# --- Where the propagation methods sample it --------------------------------------------------


def test_expm_is_exact_on_a_pwc_pulse(pwc_qubit):
    """One exponential per pulse pixel is the exact solution, so it is the reference here.

    The pulse does not vary within a pixel, so refining the step size cannot change the result.
    """
    tlist, schroedinger, _ = pwc_qubit
    dt = tlist[1] - tlist[0]

    one_step = Expm(eom_func=schroedinger.get_value, resolution=1 / dt, initial_state=INITIAL_PROPAGATOR)
    forty_steps = Expm(eom_func=schroedinger.get_value, resolution=40 / dt, initial_state=INITIAL_PROPAGATOR)

    np.testing.assert_allclose(one_step.get_value(tlist), forty_steps.get_value(tlist), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("method", ["vern7", "diffrax", "chebyshev"])
@pytest.mark.parametrize("steps_per_pixel", [1, 2, 4])
def test_propagation_is_exact_on_an_aligned_pwc_pulse(pwc_qubit, method, steps_per_pixel):
    """Every propagation method reproduces ``Expm`` when its steps lie within a pulse pixel."""
    tlist, schroedinger, _ = pwc_qubit
    dt = tlist[1] - tlist[0]
    resolution = steps_per_pixel / dt

    reference = Expm(eom_func=schroedinger.get_value, resolution=resolution, initial_state=INITIAL_PROPAGATOR)
    propagation = make_propagation(method, schroedinger.get_value, resolution, INITIAL_PROPAGATOR)

    np.testing.assert_allclose(propagation.get_value(tlist), reference.get_value(tlist), rtol=0.0, atol=1e-11)


def test_every_stage_of_a_step_reads_the_pulse_pixel_of_that_step(pwc_qubit):
    """No sample of a step may fall onto the neighboring pulse pixel.

    Checked directly because one stage on the wrong side of a boundary results in large error in propagation.
    """
    tlist, schroedinger, generator = pwc_qubit
    dt = tlist[1] - tlist[0]
    propagation = Vern7(
        eom_func=schroedinger.get_value,
        resolution=1 / dt,
        initial_state=INITIAL_PROPAGATOR,
        step_function=schrodinger_step,
    )

    step_times, step_size = construct_batched_times(tlist, propagation.resolution)
    sample_times = jax.vmap(propagation._construct_time_grid, in_axes=(0, None))(step_times, step_size)

    pixel_amplitudes = np.asarray(generator.get_value(generator._time_grid))
    for pixel, times_of_pixel in enumerate(sample_times):
        sampled = np.asarray(generator.get_value(jnp.reshape(times_of_pixel, (-1,))))
        np.testing.assert_allclose(
            sampled,
            pixel_amplitudes[pixel],
            rtol=1e-15,
            atol=0.0,
            err_msg=f"a stage of step {pixel} sampled a different pulse pixel",
        )


@pytest.mark.parametrize("solver", [Vern7, DiffraxODE])
@pytest.mark.parametrize("t_start", [0.0, 5e-9, 1.0])
def test_the_step_offset_stays_between_the_rounding_and_the_step(solver, t_start):
    """The offset has to clear the spacing of the floats and stay negligible against the step.

    Checked over grids whose times are far larger than a step, where the two bounds come closest,
    and over both solvers, which carry their own copy of the derivation.
    """
    step_times = jnp.asarray(np.linspace(t_start, t_start + T_FINAL, N_PIXELS + 1)[:-1])
    dt = float(step_times[1] - step_times[0])

    offset = float(solver._step_offset(step_times, dt))

    # Large enough that a sample moved by it is a different float, at the magnitude of the times.
    largest = float(jnp.abs(step_times[-1])) + dt
    assert offset > np.spacing(largest)
    # And small enough to leave the stages of a step well inside it.
    assert offset < 1e-4 * dt
