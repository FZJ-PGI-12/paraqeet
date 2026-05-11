import jax.numpy as jnp
from jax import jit, vjp, vmap

from paraqeet.model.utils import matrix_sqrt_psd
from paraqeet.propagation.utils import convert_vec_to_dm
from paraqeet.quantity import Array


@jit
def overlap_state_vector(final_state: Array, target_state: Array, *args, **kwargs) -> Array:
    """Compute the overlap of state vectors."""
    return jnp.vdot(target_state, final_state)


@jit
def overlap_density_matrix(final_state: Array, target_state: Array, *args, **kwargs) -> Array:
    """Compute the overlap of density matrices for a pure target_state."""
    return jnp.linalg.trace(jnp.matmul(target_state, final_state))


@jit
def overlap_vectorized_density_matrix(final_state: Array, target_state: Array, *args, **kwargs) -> Array:
    """Compute the overlap of density matrices for a pure target_state."""
    target_state = convert_vec_to_dm(target_state)
    final_state = convert_vec_to_dm(final_state)
    return jnp.linalg.trace(jnp.matmul(target_state, final_state))


@jit
def overlap_density_matrix_mixed_states(final_state: Array, target_state: Array, *args, **kwargs) -> Array:
    """Compute the overlap of density matrices."""
    return jnp.linalg.trace(matrix_sqrt_psd(jnp.matmul(target_state, final_state))) ** 2


@jit
def overlap_vectorized_density_matrix_mixed_states(final_state: Array, target_state: Array, *args, **kwargs) -> Array:
    """Compute the overlap of density matrices."""
    target_state = convert_vec_to_dm(target_state)
    final_state = convert_vec_to_dm(final_state)
    return jnp.linalg.trace(matrix_sqrt_psd(jnp.matmul(target_state, final_state))) ** 2


def vjp_jacobian(f):
    """Returns a function that computes the Jacobian of f w.r.t. its first arg via vjp."""

    @jit
    def jac_fn(x, *args, **kwargs):
        # Fix all the values except the first
        f_first = lambda x_: f(x_, *args, **kwargs)
        y, vjp_fn = vjp(f_first, x)

        # For scalar outputs
        if y.ndim == 0:
            return vjp_fn(jnp.ones_like(y))[0]

        # For Array outputs
        I = jnp.eye(y.size).reshape((y.size,) + y.shape)
        jac_flat = vmap(vjp_fn)(I)[0]
        return jac_flat.reshape(y.shape + x.shape)

    return jac_fn
