"""Test GOAT over GRAPE. This is same as the example OC_GOAToverGRAPE_TLS"""

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.scipy.special import erf
import pytest

from paraqeet.quantity import Array, Quantity
from paraqeet.signal.envelopes import Envelope
from paraqeet.signal.pwc_generator import PWCGenerator
from paraqeet.model.closed_system import ClosedSystem
from paraqeet.model.rotating_frame_drive import RotatingFrameDrive
from paraqeet.model.hamiltonian import Hamiltonian
from paraqeet.measurement.state_transfer_fidelity import StateTransferFidelityGRAPE
from paraqeet.propagation.scipy_expm_grape import ScipyExpmGRAPE
from paraqeet.optimisation_map import OptimisationMap
from paraqeet.optimisers.scipy_optimiser_gradient import ScipyOptimiserGradient
from paraqeet.measurement.goat_over_grape import GOATOverGRAPE


t_final = 20e-9
tlist = jnp.linspace(0, t_final, 26)


class FlatTopGaussianEnvelope(Envelope):
    """A flat-top Gaussian envelope."""

    def __init__(
        self,
        amplitude: Quantity,
        t_up: Quantity,
        t_down: Quantity,
        ramp_time: Quantity,
    ):
        self._amplitude = amplitude
        self.__t_up = t_up
        self.__t_down = t_down
        self.__ramp_time = ramp_time

        self._gradient_function: Callable | None = None
        self._grad_arg_nums: tuple[int, ...] = ()

    def get_parameters(self):
        """Get all parameters of the system."""
        return [self._amplitude, self.__t_up, self.__t_down, self.__ramp_time]

    @partial(jit, static_argnums=(0,))
    def _evaluate(self, amp: Array, t_up: Array, t_down: Array, ramp_time: Array, t: Array):
        rampUp = 1 + erf((t - t_up) / ramp_time)
        rampDown = 1 + erf((-t + t_down) / ramp_time)
        return jnp.squeeze(amp * rampUp * rampDown / 4)

    def compute_output(self, t: Array) -> Array:
        """Compute pulse shape."""
        amp = self._amplitude.get_value()
        t_up = self.__t_up.get_value()
        t_down = self.__t_down.get_value()
        ramp_time = self.__ramp_time.get_value()
        return self._evaluate(amp, t_up, t_down, ramp_time, t)


class SpinRWA(Hamiltonian):
    """A Single Spin."""

    def __init__(self, drives=None):
        super().__init__(drives)
        self.sigma_p = jnp.array([[0j, 1], [0, 0]])
        self.dim = 2

    def get_matrix_one_time(self, t):
        """Just sigma-X."""
        return self._drives[0].get_matrix_one_time(self.sigma_p, t)

    def gradient(self, t):
        """Gradient is just the drive matrix."""
        return self._drives[0].gradient(self.sigma_p, t)


@pytest.fixture
def tone():
    tone = FlatTopGaussianEnvelope(
        amplitude=Quantity(jnp.pi / t_final / 3, -jnp.pi / t_final, jnp.pi / t_final, name="Amplitude"),
        t_up=Quantity(1e-9, 0.0, t_final, name="t_up"),
        t_down=Quantity(t_final - 1e-9, 0.0, t_final, name="t_down"),
        ramp_time=Quantity(2e-9, 0.5e-9, t_final, name="ramp_time"),
    )
    return tone


@pytest.fixture
def gen(tone):
    gen = PWCGenerator(envelopes=[tone], tlist=tlist)
    gen.multiply_flat_top = True

    params = gen.get_parameters()
    params[0].set_limits(-200e6, 200e6)
    params[1].set_limits(-200e6, 200e6)
    return gen


@pytest.fixture
def model(gen):
    drive = RotatingFrameDrive(gen)
    spin = SpinRWA(drives=[drive])
    model = ClosedSystem(spin)
    return model


@pytest.fixture
def fid(model):
    prop = ScipyExpmGRAPE(model, res=1e9)

    init = jnp.array([[1.0], [0]])  # |0>
    target = jnp.array([[0.0], [1]])  # |1>

    prop.set_initial_state(init)
    prop.target_state = target

    zeroone = StateTransferFidelityGRAPE(
        propagation=prop,
        initial_state=init,
        target_state=target,
        times=tlist,
    )
    return zeroone


@pytest.fixture
def optGrad(tone, fid, gen):
    optmap = OptimisationMap()
    optmap.add(tone)
    optmap.register_params_with_optimisables()

    goat = GOATOverGRAPE(fid, gen)
    optGrad = ScipyOptimiserGradient(goat, optimisables=optmap)
    return optGrad


def test_goat_over_grape(optGrad):
    res = optGrad.optimise()
    assert res.value < 1e-4
