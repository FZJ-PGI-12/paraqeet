import jax.numpy as jnp
from jax import jit

from paraqeet.propagation.utils import convert_vec_to_dm


@jit
def overlap_state_vector(target_state, final_state):
    """Compute the overlap of state vectors."""
    return jnp.vdot(target_state, final_state)


@jit
def overlap_density_matrix(target_state, final_state):
    """Compute the overlap of density matrices."""
    # TODO: Make sure this works for mixed states: implement fidelity using sqrt(rho)
    return jnp.linalg.trace(jnp.matmul(target_state, final_state))


def overlap_vectorized_density_matrix(target_state, final_state):
    """Compute the overlap of density matrices."""
    # TODO: Make sure this works for mixed states: implement fidelity using sqrt(rho)
    dim = jnp.sqrt(target_state.shape[0]).astype(int)
    target_state = convert_vec_to_dm(target_state, dim)
    final_state = convert_vec_to_dm(final_state, dim)
    return jnp.linalg.trace(jnp.matmul(target_state, final_state))
