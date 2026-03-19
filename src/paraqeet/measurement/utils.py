import jax.numpy as jnp


def overlap_state_vector(target_state, final_state):
    return jnp.vdot(target_state, final_state)


def overlap_density_matrix(target_state, final_state):
    # TODO: Make sure this works for mixed states
    return jnp.linalg.trace(jnp.matmul(target_state, final_state))
