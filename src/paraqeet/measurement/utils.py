"""Utility functions for measurements, such as state overlaps and Jacobian helpers."""

import jax.numpy as jnp
from jax import jit

from paraqeet.hamiltonian.utils import matrix_sqrt_psd
from paraqeet.propagation.utils import convert_vec_to_dm
from paraqeet.quantity import Array


@jit
def overlap_state_vector(final_state: Array, target_state: Array) -> Array:
    """Compute the overlap of state vectors."""
    return jnp.vdot(target_state, final_state)


@jit
def overlap_density_matrix(final_state: Array, target_state: Array) -> Array:
    """Compute the overlap of density matrices for a pure target_state."""
    return jnp.linalg.trace(jnp.matmul(target_state, final_state))


@jit
def overlap_vectorized_density_matrix(final_state: Array, target_state: Array) -> Array:
    """Compute the overlap of density matrices for a pure target_state."""
    target_state = convert_vec_to_dm(target_state)
    final_state = convert_vec_to_dm(final_state)
    return jnp.linalg.trace(jnp.matmul(target_state, final_state))


@jit
def overlap_density_matrix_mixed_states(final_state: Array, target_state: Array) -> Array:
    """Compute the overlap of density matrices."""
    return jnp.linalg.trace(matrix_sqrt_psd(jnp.matmul(target_state, final_state))) ** 2


@jit
def overlap_vectorized_density_matrix_mixed_states(final_state: Array, target_state: Array) -> Array:
    """Compute the overlap of density matrices."""
    target_state = convert_vec_to_dm(target_state)
    final_state = convert_vec_to_dm(final_state)
    return jnp.linalg.trace(matrix_sqrt_psd(jnp.matmul(target_state, final_state))) ** 2


@jit
def gate_fidelity(overlap: Array) -> Array:
    """Compute the fidelity of a gate based on state overlaps."""
    return jnp.abs(jnp.average(overlap)) ** 2


@jit
def state_fidelity(overlap: Array) -> Array:
    """Compute the average state transfer fidelity based on state overlaps."""
    return jnp.average(jnp.abs(overlap) ** 2)
