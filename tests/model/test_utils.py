"""Test model utils functions."""

import jax.numpy as jnp
import numpy as np
import pytest

from paraqeet.model.utils import (
    construct_annihilation_op,
    construct_composite_basis_state,
    construct_creation_op,
    convert_state_to_dm,
    identity_operator,
    matrix_sqrt_psd,
    partial_trace,
    sigma_minus,
    sigma_plus,
    sigma_x,
    sigma_y,
    sigma_z,
    tensor,
)


# Test common operators
def test_sigma_x():
    """Return the Pauli-X operator."""
    assert np.all(sigma_x() == np.array([[0.0j, 1.0], [1.0, 0.0j]]))


def test_sigma_y():
    """Return the Pauli-Y operator."""
    assert np.all(sigma_y() == np.array([[0.0, 1.0j], [-1.0j, 0.0]]))


def test_sigma_z():
    """Return the Pauli-Z operator."""
    assert np.all(sigma_z() == np.array([[1.0, 0.0j], [0.0j, -1.0]]))


def test_sigma_plus():
    """Return the Pauli-creation operator."""
    assert np.all(sigma_plus() == np.array([[0.0j, 1.0], [0.0, 0.0j]]))


def test_sigma_minus():
    """Return the Pauli-annihilation operator."""
    assert np.all(sigma_minus() == np.array([[0.0j, 0.0], [1.0, 0.0j]]))


@pytest.mark.parametrize("dim", [2, 5, 10])
def test_annihilation_operator(dim):
    assert np.all(construct_annihilation_op(dim) == jnp.diag(jnp.sqrt(jnp.arange(1, dim, dtype=jnp.complex128)), k=1))


@pytest.mark.parametrize("dim", [2, 5, 10])
def test_creation_operator(dim):
    assert np.all(construct_creation_op(dim) == jnp.diag(jnp.sqrt(jnp.arange(1, dim, dtype=jnp.complex128)), k=-1))


@pytest.mark.parametrize("dim", [2, 5, 10])
def test_identity_operator(dim: int):
    """Return the identity operator for the specified dimensions."""
    assert np.all(identity_operator(dim) == np.eye(dim))


# Tests for matrix sqrt for positive semi-definite operators
def test_identity():
    """sqrt(Identity) = Identity."""
    Identity = jnp.eye(5)
    np.testing.assert_allclose(matrix_sqrt_psd(Identity), Identity, atol=1e-6)


def test_diagonal():
    """sqrt(diag(d)) = diag(sqrt(d))."""
    d = jnp.array([1.0, 4.0, 9.0, 16.0])
    np.testing.assert_allclose(matrix_sqrt_psd(jnp.diag(d)), jnp.diag(jnp.sqrt(d)), atol=1e-6)


@pytest.mark.parametrize("n", [2, 5, 10])
def test_square_of_sqrt(n):
    """S @ S == A for Hermitian positive semi-definite A."""
    rng = np.random.default_rng(0)
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = jnp.asarray(M @ M.conj().T)
    S = matrix_sqrt_psd(A)
    np.testing.assert_allclose(S @ S, A, atol=1e-5, rtol=1e-5)


# Tests for partial trace


@pytest.mark.parametrize("dim_A,dim_B", [(2, 3), (5, 2), (10, 5)])
def test_partial_trace_product_state(random_state, dim_A, dim_B):
    r"""Partial trace of ρ_A $\otimes$ ρ_B returns ρ_A or ρ_B."""
    psi_A = random_state(dim_A)
    psi_B = random_state(dim_B)
    rho_A = convert_state_to_dm(psi_A)
    rho_B = convert_state_to_dm(psi_B)
    rho = tensor(rho_A, rho_B)
    np.testing.assert_allclose(partial_trace(rho, (dim_A, dim_B), (0,)), rho_A, atol=1e-5)
    np.testing.assert_allclose(partial_trace(rho, (dim_A, dim_B), (1,)), rho_B, atol=1e-5)


def test_trace_out_all_gives_total_trace(random_density_matrix):
    """Tracing over all subsystems gives the same result as trace of the state."""
    rho = random_density_matrix(4)
    out = partial_trace(rho, dims=(2, 2), keep=())
    np.testing.assert_allclose(out, jnp.array([[jnp.trace(rho)]]), atol=1e-6)


def test_bell_state_reduced_is_maximally_mixed():
    """Tracing either qubit of a Bell state gives I/2."""
    state_10 = construct_composite_basis_state(dims=(2, 2), index=(1, 0))
    state_01 = construct_composite_basis_state(dims=(2, 2), index=(0, 1))
    bell = (state_10 + state_01) / jnp.sqrt(2)
    rho = convert_state_to_dm(bell)
    for keep in [(0,), (1,)]:
        np.testing.assert_allclose(
            partial_trace(rho, dims=(2, 2), keep=keep),
            identity_operator(2) / 2,
            atol=1e-6,
        )
