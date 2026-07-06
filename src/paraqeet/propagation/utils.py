"""Utility functions for propagation module."""

import math

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from paraqeet.quantity import Array


@jit
def convert_dm_to_vec(state_dm: Array) -> jnp.ndarray:
    """Convert a density matrix to vectorized form."""
    dim = state_dm.shape[1]
    vec = jnp.reshape(jnp.transpose(state_dm), (-1, dim**2, 1))
    if vec.shape[0] == 1:
        vec = jnp.squeeze(vec, axis=0)
    return vec


@jit
def convert_vec_to_dm(state_vec: Array) -> jnp.ndarray:
    """Convert a vectorized density matrix to matrix form."""
    dim = math.isqrt(state_vec.shape[0])
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


def lindblad_step(state: Array, h: Array, cols: list[Array], *args, **kwargs):
    """Step function for ODE propagation methods, such as Vern7, for the Lindblad master equation."""
    del_rho = commutator(h, state)
    for col in cols:
        del_rho += jnp.matmul(jnp.matmul(col, state), dagger(col))
        del_rho -= 0.5 * anti_commutator(jnp.matmul(dagger(col), col), state)
    return del_rho


def schrodinger_step(state: Array, h: Array, *args, **kwargs):
    """Step function for ODE propagation methods, such as Vern7, for the Schrödinger equation."""
    return jnp.matmul(h, state)


def reverse_schrodinger_step(state: Array, h: Array, *args, **kwargs):
    """Reverse step function for ODE propagation methods, such as Vern7GRAPE, for the Schrödinger equation."""
    return jnp.matmul(state, h)


def reverse_lindblad_step(state: Array, h: Array, cols: list[Array], *args, **kwargs):
    """Reverse step function for ODE propagation methods, such as Vern7GRAPE, for the Lindblad master equation."""
    del_rho = commutator(h, state)
    for col in cols:
        del_rho -= jnp.matmul(jnp.matmul(dagger(col), state), col)
        del_rho += 0.5 * anti_commutator(jnp.matmul(dagger(col), col), state)
    return del_rho


@jit
def grape_operator_sandwich_function_closed(ham_grads, fwd_prop_states, rev_prop_states):
    r"""
    Operator sandwich function for GRAPE for closed system implementing
        .. math::
            \langle \lambda(t) \lvert \frac{\partial H}{\partial \alpha} \rvert \psi(t) \rangle.
    """
    fwd_multiply = vmap(jnp.matmul, in_axes=(0, 0))(ham_grads, fwd_prop_states)
    grad = vmap(jnp.matmul, in_axes=(0, 0))(rev_prop_states, fwd_multiply)
    return grad


@jit
def grape_operator_sandwich_function_open(ham_grads, fwd_prop_states, rev_prop_states):
    r"""
    Operator sandwich function for GRAPE for open system implementing
        .. math::
            \text{Tr}(\sigma(t) [H, \rho(t)]).
    """
    fwd_multiply = vmap(commutator, in_axes=(0, 0))(ham_grads, fwd_prop_states)
    grad = vmap(jnp.matmul, in_axes=(0, 0))(rev_prop_states, fwd_multiply)
    return jnp.linalg.trace(grad)


def construct_times(times: Array, ti: int, resolution: float) -> tuple:
    """Construct one-dimensional vector of time.

    Interpolate the user-specified times to match the propagation resolution.

    Args:
        times: Array of times.
        ti: Snapshot of the time at a current step.
        resolution: Time steps resolution.

    Returns:
        Array: Array of timestamps in specified resolution.
        int: Difference in time step.

    """
    t0 = times[ti - 1]
    t1 = times[ti]
    steps = int(np.floor((t1 - t0) * resolution + 0.5))
    if steps == 0:
        steps = 1
    new_times = jnp.linspace(t0, t1, steps, endpoint=False)
    if steps < 2:
        dt = t1 - t0
    else:
        dt = new_times[1] - new_times[0]
    return new_times, dt
