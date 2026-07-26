"""Utilities for model construction."""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.scipy.linalg import sqrtm

from paraqeet.quantity import Array


def sigma_x() -> Array:
    r"""Return the Pauli-X operator.
    
    .. math::

        \begin{bmatrix}
        0 & 1 \\
        1 & 0
        \end{bmatrix}
    """
    return np.array([[0.0, 1.0], [1.0, 0.0]])


def sigma_y() -> Array:
    r"""Return the Pauli-Y operator.
    
    .. math::

        \begin{bmatrix}
        0 & -i \\
        i & 0
        \end{bmatrix}
    """
    return np.array([[0.0, -1.0j], [1.0j, 0.0]])


def sigma_z() -> Array:
    r"""Return the Pauli-Z operator.
    
    .. math::

        \begin{bmatrix}
        1 & 0 \\
        0 & -1
        \end{bmatrix}
    """
    return np.array([[1.0, 0.0], [0.0, -1.0]])


def sigma_minus() -> Array:
    r"""Return the Pauli minus operator in the quantum information convention.
    
    .. math::

        \begin{bmatrix}
        0 & 1 \\
        0 & 0
        \end{bmatrix}
    """
    return np.array([[0.0, 1.0], [0.0, 0.0]])


def sigma_plus() -> Array:
    r"""Return the Pauli plus operator in the quantum information convention.

    .. math::

        \begin{bmatrix}
        0 & 0 \\
        1 & 0
        \end{bmatrix}
    """
    return np.array([[0.0, 0.0], [1.0, 0.0]])


def identity_operator(dim: int) -> Array:
    """Return the identity operator for the specified dimensions."""
    return np.eye(dim)


def dagger(op: Array) -> Array:
    r"""Return transpose conjugate of an operator :math:`O^\dagger = (O^T)^{*}`."""
    return op.T.conj()


@jit
def matrix_sqrt(op: Array) -> Array:
    """Return the matrix square root using jax based implementation.
    This works for any general matrix with positive eigenvalues.

    Uses jax.scipy.lingalg.sqrtm for the implementation.

    Note:
        This function does not support automatic-differentiation (AD).
        Use ``matrix_sqrt_psd`` for positive semi-definite matrices for using AD.
    """
    return sqrtm(op)


@jit
def matrix_sqrt_psd(a_mat: Array) -> Array:
    """Matrix square root of a Hermitian positive semi-definite matrix like a density matrix."""
    w, v_mat = jnp.linalg.eigh(a_mat)
    w_sqrt = jnp.sqrt(jnp.clip(w, min=0.0))  # clip tiny negatives from roundoff
    v_dag: Array = dagger(v_mat)
    # ignoring mypy due to jit
    return (v_mat * w_sqrt) @ v_dag  # type: ignore


def partial_trace(rho: Array, dims: tuple[int, ...], keep: tuple[int, ...]) -> Array:
    """Trace out all subsystems except those whose indices are in ``keep``.

    Args:
        rho: Density matrix, shape (D, D) with D = prod(dims).
        dims: Tuple of subsystem dimensions.
        keep: Tuple of subsystem indices to keep. Counting starts from 0.

    Returns:
        The reduced density matrix after tracing out the specified subsystems.
    """
    n = len(dims)
    # Row axes 0..n-1, column axes n..2n-1.
    # For traced subsystems, tie col axis to row axis -> einsum sums it.
    row = list(range(n))
    col = list(range(n, 2 * n))
    for i in range(n):
        if i not in keep:
            col[i] = row[i]

    out_axes = [row[i] for i in keep] + [col[i] for i in keep]

    rho_t = rho.reshape(dims + dims)
    out = jnp.einsum(rho_t, row + col, out_axes)

    d_keep = 1
    for i in keep:
        d_keep *= dims[i]
    return out.reshape(d_keep, d_keep)


def construct_annihilation_op(dim: int) -> Array:
    """Create bosonic annihilation operator for a system with dimensions ``dim``."""
    return jnp.diag(jnp.sqrt(jnp.arange(1, dim, dtype=jnp.complex128)), k=1)


def construct_creation_op(dim: int) -> Array:
    """Create bosonic creation operator for a system with dimensions ``dim``."""
    return jnp.diag(jnp.sqrt(jnp.arange(1, dim, dtype=jnp.complex128)), k=-1)


def construct_basis_state(dim: int, index: int) -> Array:
    r"""Generate pure basis state for a single system.

    Args:
        dim: Dimension of the system.
        index: Index of the state, for e.g., for fock state :math:`|0\rangle`, index = 0.
            Max index = dim - 1.

    Returns:
        Basis state corresponding to the dimension and index.
    """
    if index >= dim:
        raise Exception(f"``index`` has to be less than ``dim``. Got dim={dim}, index={index}.")

    state = np.zeros(dim)
    state[index] = 1
    return state


def construct_composite_basis_state(dims: tuple[int, ...], index: tuple[int, ...]) -> Array:
    """Generate pure Basis state of a composite system based on the index.

    Args:
        dims: Subsystem dimensions as a tuple.
        index: Index of the subsystem states as a tuple.
    """
    if len(dims) != len(index):
        raise Exception("Length of dims and index must be same")

    if np.any((np.array(dims) - np.array(index)) <= 0):
        raise Exception("Index has to be less than subsystem dimension")

    individual_states = [construct_basis_state(dim, idx) for dim, idx in zip(dims, index)]
    return np.reshape(ntensor(individual_states), (-1, 1))


def convert_state_to_dm(state: Array) -> Array:
    """Convert a pure state into a density matrix by taking the outer product."""
    return state @ state.T.conj()


@jit
def tensor(op_a: Array, op_b: Array) -> Array:
    """Tensor product of two operators."""
    return jnp.kron(op_a, op_b)


def ntensor(ops: list[Array]) -> Array:
    r"""Tensor product of a list of operators in the left to right order.

    Returns the operator:
        .. math::
            \text{ntensor}[A_1, A_2, ..., A_N] = A_1 \otimes A_2 \otimes ... \otimes A_N.
    """
    full_op = ops[0]
    for op in ops[1:]:
        full_op = tensor(full_op, op)
    return full_op


def tensor_product_with_identity(mat_list: list[Array], n: list[int], dims: list[int]) -> Array:
    r"""Put the matrices mat_list into a tensor product at positions ``n``.

    All other positions are identity matrices:
        .. math::
            1 \otimes \dots \otimes 1 \otimes \text{mat_list}_1 \otimes 1 \otimes
            \dots \otimes 1 \otimes \text{mat_list}_2 \dots

    The dimensions are assumed to be the same as the subsystems.

    Args:
        mat_list: List of Matrices for tensor product.
        n: List of indices for the each mat_list_i.

    Returns:
        Tensor product of mat_list_i's with I's.
    """
    # Create identity matrices for all subsystems and
    # fill in mat_list at the corresponding indices
    sub_matrices = [jnp.eye(dim) for dim in dims]
    for i, k in enumerate(n):
        sub_matrices[k] = jnp.array(mat_list[i])

    # Tensor product everything in sub_matrices
    product = jnp.eye(1)
    for m in sub_matrices:
        product = jnp.kron(product, m)

    return product


## Helper functions for cross-package support

# Numpy


def np_func_to_jax_func(ham_func: Callable) -> Callable[..., Array]:
    """Convert a Numpy Hamiltonian function to JAX compatible function.

    Adds ``vmap`` capabilities to vectorize the computation over a batch of times
    (first variable).

    Args:
        ham_func: Numpy based Hamiltonian function to convert to JAX and vmap
            compatible function.
    """

    def _jax_wrapper(times: Array, *args: Any, **kwargs: Any) -> Array:
        one_time_func = lambda t: jnp.array(ham_func(t, *args, **kwargs))
        return vmap(one_time_func)(times)

    return _jax_wrapper


# QuTiP
def qobj_to_array(qobj: Any) -> Array:
    """Convert a QuTiP-JAX :cite:p:`lambert2026qutip` object to a JAX array.

    Args:
        qobj: QuTiP object.
    """
    arr: Array = qobj.data._jxa
    return arr


def qt_func_to_jax_func(ham_func: Callable) -> Callable[..., Array]:
    """Convert a QuTiP-JAX Hamiltonian function to JAX compatible function.

    Adds ``vmap`` capabilities to vectorize the computation over a batch of times
    (first variable).

    Args:
        ham_func: QuTiP based Hamiltonian function to convert to JAX and vmap
            compatible function.
    """

    def _jax_wrapper(times: Array, *args: Any, **kwargs: Any) -> Array:
        one_time_func = lambda t: ham_func(t, *args, **kwargs).data._jxa
        values: Array = vmap(one_time_func)(times)
        return values

    return _jax_wrapper
