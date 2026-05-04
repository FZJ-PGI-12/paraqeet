import jax.numpy as jnp
from jax import jit, vmap

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


@jit
def commutator(A: Array, B: Array):
    """Compute the commutator between two operators A and B."""
    return jnp.matmul(A, B) - jnp.matmul(B, A)


@jit
def anti_commutator(A: Array, B: Array):
    """Compute the anti-commutator between two operators A and B."""
    return jnp.matmul(A, B) + jnp.matmul(B, A)


@jit
def dagger(op: Array):
    """Compute the dagger of an operator."""
    return op.conj().T


def lindblad_step(state: Array, h: Array, cols: list[Array], **kwargs):
    """Step function for ODE propagation methods, such as Vern7, for the Lindblad master equation."""
    del_rho = commutator(h, state)
    for col in cols:
        del_rho += jnp.matmul(jnp.matmul(col, state), dagger(col))
        del_rho -= 0.5 * anti_commutator(jnp.matmul(dagger(col), col), state)
    return del_rho


def schrodinger_step(state: Array, h: Array, **kwargs):
    """Step function for ODE propagation methods, such as Vern7, for the Schrödinger equation."""
    return jnp.matmul(h, state)


def reverse_schrodinger_step(state: Array, h: Array, **kwargs):
    """Reverse step function for ODE propagation methods, such as Vern7GRAPE, for the Schrödinger equation."""
    return jnp.matmul(state, h)


def reverse_lindblad_step(state: Array, h: Array, cols: list[Array], **kwargs):
    """Reverse step function for ODE propagation methods, such as Vern7GRAPE, for the Lindblad master equation."""
    del_rho = commutator(h, state)
    for col in cols:
        del_rho -= jnp.matmul(jnp.matmul(dagger(col), state), col)
        del_rho += 0.5 * anti_commutator(jnp.matmul(dagger(col), col), state)
    return del_rho


@jit
def grape_operator_sandwich_function_closed(ham_grads, fwd_prop_states, rev_prop_states):
    """Operator sandwich function for GRAPE for closed system implementing
    .. math::
            \\langle \\lambda(t) \\lvert \\frac{\\partial H}{\\partial \\alpha} \\rvert \\psi(t) \\rangle
    """
    fwd_multiply = vmap(jnp.matmul, in_axes=(0, 0))(ham_grads, fwd_prop_states)
    grad = vmap(jnp.matmul, in_axes=(0, 0))(rev_prop_states, fwd_multiply)
    return grad


@jit
def grape_operator_sandwich_function_open(ham_grads, fwd_prop_states, rev_prop_states):
    """Operator sandwich function for GRAPE for open system implementing
    .. math::
            \\text{Tr}(\\sigma(t) [H, \\rho(t)])
    """
    fwd_multiply = vmap(commutator, in_axes=(0, 0))(ham_grads, fwd_prop_states)
    grad = vmap(jnp.matmul, in_axes=(0, 0))(rev_prop_states, fwd_multiply)
    return grad
