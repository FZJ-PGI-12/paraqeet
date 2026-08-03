"""Utility functions for propagation module."""

import math
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from paraqeet.quantity import Array, Float


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
def commutator(A: Array, B: Array) -> Array:
    """Compute the commutator between two operators A and B."""
    return jnp.matmul(A, B) - jnp.matmul(B, A)


@jit
def anti_commutator(A: Array, B: Array) -> Array:
    """Compute the anti-commutator between two operators A and B."""
    return jnp.matmul(A, B) + jnp.matmul(B, A)


@jit
def dagger(op: Array) -> Array:
    """Compute the dagger of an operator."""
    return op.conj().T


def lindblad_step(state: Array, h: Array, cols: Array, *args: Any, **kwargs: Any) -> Array:
    """Step function for ODE propagation methods, such as Vern7, for the Lindblad master equation."""
    del_rho: Array = commutator(h, state)
    for col in cols:
        del_rho += jnp.matmul(jnp.matmul(col, state), dagger(col))
        del_rho -= 0.5 * anti_commutator(jnp.matmul(dagger(col), col), state)
    return del_rho


def schrodinger_step(state: Array, h: Array, *args: Any, **kwargs: Any) -> Array:
    """Step function for ODE propagation methods, such as Vern7, for the Schrödinger equation."""
    return jnp.matmul(h, state)


def reverse_lindblad_step(state: Array, h: Array, cols: Array, *args: Any, **kwargs: Any) -> Array:
    r"""Backward step function of :class:`~paraqeet.propagation.grape.GRAPE` for the Lindblad master equation.

    Implements the adjoint Lindbladian

        .. math::
            \mathcal{L}^\dagger(\sigma) = i[H, \sigma]
            + \sum_k L_k^\dagger \sigma L_k - \frac{1}{2}\{L_k^\dagger L_k, \sigma\},

    which is the right hand side of the backward propagation of the target state in reverse time.
    Compared to the forward ``lindblad_step`` the coherent part changes sign while the dissipator
    keeps its signs and only exchanges the collapse operators with their adjoints. The sign of the
    coherent part is already taken care of by the adjoint EOM that ``GRAPE`` passes in, so ``h`` is
    :math:`iH\,\mathrm{d}t` here and the commutator has the same form as in the forward step.
    """
    del_sigma: Array = commutator(h, state)
    for col in cols:
        del_sigma += jnp.matmul(jnp.matmul(dagger(col), state), col)
        del_sigma -= 0.5 * anti_commutator(jnp.matmul(dagger(col), col), state)
    return del_sigma


@jit
def grape_operator_sandwich_function_closed(ham_grads: Array, fwd_prop_states: Array, rev_prop_states: Array) -> Array:
    r"""
    Operator sandwich function for GRAPE for closed system implementing
        .. math::
            \langle \lambda(t) \lvert \frac{\partial H}{\partial \alpha} \rvert \psi(t) \rangle.
    """
    fwd_multiply = vmap(jnp.matmul, in_axes=(0, 0))(ham_grads, fwd_prop_states)
    grad = vmap(jnp.matmul, in_axes=(0, 0))(rev_prop_states, fwd_multiply)
    return grad


@jit
def grape_operator_sandwich_function_open(ham_grads: Array, fwd_prop_states: Array, rev_prop_states: Array) -> Array:
    r"""
    Operator sandwich function for GRAPE for open system implementing
        .. math::
            \text{Tr}(\sigma(t) [H, \rho(t)]).
    """
    fwd_multiply = vmap(commutator, in_axes=(0, 0))(ham_grads, fwd_prop_states)
    grad = vmap(jnp.matmul, in_axes=(0, 0))(rev_prop_states, fwd_multiply)
    return jnp.linalg.trace(grad)


def construct_times(times: Array, ti: int, resolution: float) -> tuple[Array, Float]:
    """Construct one-dimensional vector of time.

    Interpolate the user-specified times to match the propagation resolution.

    Args:
        times: Array of times.
        ti: Index of the current step into ``times``; the interval [times[ti - 1], times[ti]) is interpolated.
        resolution: Time steps resolution.

    Returns:
        Array: Array of timestamps in specified resolution.
        Float: Difference between two consecutive time steps.

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
