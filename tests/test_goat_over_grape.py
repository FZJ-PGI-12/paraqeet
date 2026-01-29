"""Test GOAT over GRAPE. This is same as the example OC_GOAToverGRAPE_TLS"""

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import numpy as np
import numpy.testing as testing
import pytest
from jax import jit
from jax.scipy.special import erf

from paraqeet.measurement.goat_over_grape import GOATOverGRAPE
from paraqeet.measurement.state_transfer_fidelity import (
    StateTransferFidelityGRAPE,
)
from paraqeet.model.closed_system import ClosedSystem
from paraqeet.model.differentiable_hamiltonian import DifferentiableHamiltonian
from paraqeet.model.rotating_frame_drive import RotatingFrameDrive
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.scipy_optimizer_gradient import ScipyOptimizerGradient
from paraqeet.propagation.scipy_expm_grape import ScipyExpmGRAPE
from paraqeet.quantity import Array, Quantity
from paraqeet.signal.envelopes import Envelope
from paraqeet.signal.pwc_generator import PWCGenerator

T_FINAL = 20e-9
TLIST = jnp.linspace(0, T_FINAL, 26)


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
        ramp_up = 1 + erf((t - t_up) / ramp_time)
        ramp_down = 1 + erf((-t + t_down) / ramp_time)
        return jnp.squeeze(amp * ramp_up * ramp_down / 4)

    def get_value(self, t: Array | float) -> Array:
        """Compute pulse shape."""
        amp = self._amplitude.get_value()
        t_up = self.__t_up.get_value()
        t_down = self.__t_down.get_value()
        ramp_time = self.__ramp_time.get_value()
        # returns JitWrapped
        return self._evaluate(amp, t_up, t_down, ramp_time, t) # type: ignore


class SpinRWA(DifferentiableHamiltonian):
    """A Single Spin."""

    def __init__(self, drives=None):
        super().__init__(drives)
        self.sigma_p = jnp.array([[0j, 1], [0, 0]])
        self.dim = 2

    def get_value_at_timestep(self, timestep: float) -> Array:
        """Just sigma-X."""
        return self._drives[0].value_at_timestep(self.sigma_p, timestep)

    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
        """Gradient is just the drive matrix."""
        return self.get_value(times), self._drives[0].get_gradient(self.sigma_p, times)

    def get_gradient_at_timestep(self, time):
        return self._drives[0].get_gradient_at_timestep(self.sigma_p, time)

    # TODO: implement dimension-method from Hamiltonian
    def dimension(self) -> int:
        raise NotImplementedError()

    #  TODO: implement get_collapseops method from Hamiltonian
    def get_collapseops(self) -> list[tuple[Array, Array]]:
        raise NotImplementedError()

    #  TODO: implement get_parameters method from Optimizable
    def get_parameters(self):
        raise NotImplementedError()


@pytest.fixture
def tone():
    tone = FlatTopGaussianEnvelope(
        amplitude=Quantity(jnp.pi / T_FINAL / 3, -jnp.pi / T_FINAL, jnp.pi / T_FINAL, name="Amplitude"),
        t_up=Quantity(1e-9, 0.0, T_FINAL, name="t_up"),
        t_down=Quantity(T_FINAL - 1e-9, 0.0, T_FINAL, name="t_down"),
        ramp_time=Quantity(2e-9, 0.5e-9, T_FINAL, name="ramp_time"),
    )
    return tone


@pytest.fixture
def gen(tone):
    gen = PWCGenerator(envelopes=[tone], tlist=TLIST)
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
    )
    return zeroone


@pytest.fixture
def opt_grad(tone, fid, gen):
    optmap = OptimizationMap()
    optmap.add(tone)
    optmap.register_params_with_optimizables()

    goat = GOATOverGRAPE(fid, gen)
    opt_grad = ScipyOptimizerGradient(goat, optimizables=optmap)
    return opt_grad


def test_can_measure(fid, gen):
    fid = GOATOverGRAPE(fid, gen)
    val, grad = fid.get_value_and_gradient(times=TLIST)
    assert 0 <= fid.measure(times=TLIST)
    assert 0 <= val <= 1

    value = fid.calculate_normalized_scalar(times=TLIST)
    assert 0 <= value
    testing.assert_array_less(np.zeros_like(grad), grad)


def test_goat_over_grape(opt_grad):
    res = opt_grad.optimize(times=TLIST)
    assert res.value < 1e-4
