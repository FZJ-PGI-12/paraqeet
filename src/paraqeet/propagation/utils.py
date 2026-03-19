import jax.numpy as jnp

from paraqeet.quantity import Array


def convert_dm_to_vec(state_dm: Array, dim: int) -> jnp.ndarray:
    """Helper function to convert a density matrix to vectorized form."""
    vec = jnp.reshape(jnp.transpose(state_dm), (-1, dim**2, 1))
    if vec.shape[0] == 1:
        vec = jnp.squeeze(vec, axis=0)
    return vec


def convert_vec_to_dm(state_vec: Array, dim: int) -> jnp.ndarray:
    """Helper function to convert a Vectorized density matrix to matrix form."""
    dm = jnp.reshape(state_vec, (-1, dim, dim))
    if dm.shape[0] == 1:
        dm = jnp.squeeze(dm, axis=0)
    return jnp.transpose(dm)
