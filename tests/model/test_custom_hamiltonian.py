import jax.numpy as jnp
import numpy.random as random
import pytest

from paraqeet.measurement.state_transfer_fidelity import StateTransferFidelity
from paraqeet.model.closed_system import ClosedSystem
from paraqeet.model.custom_hamiltonian import CustomHamiltonian
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.scipy_optimizer_gradient import ScipyOptimizerGradient
from paraqeet.propagation.scipy_expm_goat import ScipyExpmGOAT
from paraqeet.quantity import Array, Quantity

sigma_x = jnp.array([[0j, 1], [1, 0]])
sigma_z = jnp.diag(jnp.array([1.0, -1.0]))
FREQ = 4.8e9 * 2 * jnp.pi
t_final = 10e-9


amplitude = Quantity(
    value=jnp.array(1.55e8),
    min_value=jnp.array(0.0),
    max_value=jnp.array(1e9),
    unit="Hz",
    name="Amplitude",
    two_pi=True,
)

frequency = Quantity(
    value=jnp.array(4.8e9 * 2 * jnp.pi),
    min_value=jnp.array(0.8 * 4.8e9 * 2 * jnp.pi),
    max_value=jnp.array(1.2 * 4.8e9 * 2 * jnp.pi),
    unit="Hz",
    name="lo_freq",
    two_pi=True,
)


@pytest.fixture
def cos_envelope():
    """Define a cosine envelope."""

    def _method(t, amp, freq):
        return amp * jnp.cos(freq * t)

    return _method


@pytest.fixture
def tls_hamiltonian(cos_envelope):
    """Define a Two level system Hamiltonian."""

    def _method(t, amp, freq):
        return 0.5 * FREQ * sigma_z + sigma_x * cos_envelope(t, amp, freq)

    return _method


@pytest.fixture
def gradient_functions():
    """Define the gradient of Hamiltonian wrt amp and freq."""

    def grad_amp(t, amp, freq):
        """Gradient of Hamiltonian wrt amplitude."""
        return sigma_x * jnp.cos(freq * t)

    def grad_frequency(t, amp, freq):
        """Gradient of Hamiltonian wrt frequency."""
        return -sigma_x * amp * t * jnp.sin(freq * t)

    analytical_grad_funcs = [grad_amp, grad_frequency]
    return analytical_grad_funcs


@pytest.fixture
def tls(tls_hamiltonian, gradient_functions):
    """Closed system model."""
    tls_hamil = CustomHamiltonian(
        hamiltonian_function=tls_hamiltonian, parameters=[amplitude, frequency], gradient_functions=gradient_functions
    )
    return tls_hamil


@pytest.fixture
def fid(tls):
    """Get the fidelity function"""
    model = ClosedSystem(tls)
    prop = ScipyExpmGOAT(model, res=100e9)

    init = jnp.array([[1.0], [0]])  # |0>
    target = jnp.array([[0.0], [1]])  # |1>
    zeroone = StateTransferFidelity(
        propagation=prop,
        initial_state=init,
        target_state=target,
        times=jnp.array([0.0, t_final]),
    )
    return zeroone


@pytest.fixture
def opt(tls, fid):
    optmap = OptimizationMap()
    optmap.add(tls)
    opt = ScipyOptimizerGradient(fid, optimizables=optmap)
    return opt


def test_optimization(opt):
    """Test gradient based optimization."""
    res = opt.optimize()
    assert res.value < 1e-4


def test_shapes(random_matrix):
    for dim in range(2, 10):

        def generator(time) -> Array:
            mat = random_matrix(dim, dim)
            return mat + jnp.conjugate(mat).T

        hamil = CustomHamiltonian(hamiltonian_function=generator, parameters=[])
        assert hamil.dimension() == dim

        times = jnp.linspace(0, random.randint(1, 100) * random.random(), random.randint(2, 20))
        assert hamil.get_matrix_one_time(times[-1]).shape == (dim, dim)
        assert hamil.get_matrix(times).shape == (len(times), dim, dim)
