"""Test helper functions in autograd_utils."""

import jax.numpy as jnp
import numpy as np

from paraqeet.autograd_utils import get_value_and_jacobian_fwd, get_value_and_jacobian_rev
from paraqeet.quantity import Array, Quantity

tls_freq = 4.8e9 * 2 * jnp.pi
sigma_x = jnp.expand_dims(jnp.array([[0j, 1], [1, 0]]), axis=0)
sigma_z = jnp.expand_dims(jnp.diag(jnp.array([1.0, -1.0])), axis=0)

amplitude = Quantity(
    value=jnp.array(1.55e8),
    min_value=jnp.array(0.0),
    max_value=jnp.array(5 * 1e8),
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


def tls_hamiltonian(t: Array, amp, freq):
    """Define a Two level system Hamiltonian."""
    return 0.5 * tls_freq * sigma_z + sigma_x * amp * jnp.cos(freq * t).reshape((-1, 1, 1))


def analytical_grad_amp(t: Array, freq):
    """Gradient of Hamiltonian wrt amplitude."""
    return sigma_x * (jnp.cos(freq * t)).reshape((-1, 1, 1))


def analytical_grad_frequency(t: Array, amp, freq):
    """Gradient of Hamiltonian wrt frequency."""
    return -1 * sigma_x * (amp * t * jnp.sin(freq * t)).reshape((-1, 1, 1))


def analytical_value_and_grad(t: Array, amp, freq):
    ham_grads = jnp.stack([analytical_grad_amp(t, freq), analytical_grad_frequency(t, amp, freq)], axis=1)
    return tls_hamiltonian(t, amp, freq), ham_grads


def test_hamiltonian_autodiff_reverse():
    ts = jnp.linspace(0.0, 1.0, 10)
    freq = frequency.get_value()
    amp = amplitude.get_value()

    analytical_value, analytical_grads = analytical_value_and_grad(ts, amp, freq)
    AD_value, AD_grads = get_value_and_jacobian_rev(tls_hamiltonian, argnums=(1, 2))(ts, amp, freq)

    assert np.allclose(AD_value, analytical_value)
    assert np.allclose(jnp.squeeze(AD_grads[0]), analytical_grads[:, 0, ...])
    assert np.allclose(jnp.squeeze(AD_grads[1]), analytical_grads[:, 1, ...])


def test_hamiltonian_autodiff_forward():
    ts = jnp.linspace(0.0, 1.0, 10)
    freq = frequency.get_value()
    amp = amplitude.get_value()

    analytical_value, analytical_grads = analytical_value_and_grad(ts, amp, freq)
    AD_value, AD_grads = get_value_and_jacobian_fwd(tls_hamiltonian, argnums=(1, 2))(ts, amp, freq)

    assert np.allclose(AD_value, analytical_value)
    assert np.allclose(jnp.squeeze(AD_grads[0]), analytical_grads[:, 0, ...])
    assert np.allclose(jnp.squeeze(AD_grads[1]), analytical_grads[:, 1, ...])


def test_forward_matches_reverse():
    ts = jnp.linspace(0.0, 1.0, 10)
    freq = frequency.get_value()
    amp = amplitude.get_value()

    fwd_value, fwd_grads = get_value_and_jacobian_fwd(tls_hamiltonian, argnums=(1, 2))(ts, amp, freq)
    rev_value, rev_grads = get_value_and_jacobian_rev(tls_hamiltonian, argnums=(1, 2))(ts, amp, freq)

    assert np.allclose(fwd_value, rev_value)
    assert np.allclose(fwd_grads[0], rev_grads[0])
    assert np.allclose(fwd_grads[1], rev_grads[1])


def test_forward_int_argnums():
    ts = jnp.linspace(0.0, 1.0, 10)
    freq = frequency.get_value()
    amp = amplitude.get_value()

    # A single int argnums should return a bare Jacobian, not a tuple.
    _, amp_grad = get_value_and_jacobian_fwd(tls_hamiltonian, argnums=1)(ts, amp, freq)

    assert not isinstance(amp_grad, tuple)
    assert np.allclose(jnp.squeeze(amp_grad), jnp.squeeze(analytical_grad_amp(ts, freq)))
