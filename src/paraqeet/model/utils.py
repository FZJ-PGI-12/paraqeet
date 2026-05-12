"""Utilies for model construction."""

from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.scipy.linalg import sqrtm

from paraqeet.model.system import OpenSystem
from paraqeet.quantity import Array


def sigma_x():
    """Return the Pauli-X operator."""
    return np.array([[0.0j, 1.0], [1.0, 0.0j]])


def sigma_y():
    """Return the Pauli-Y operator."""
    return np.array([[0.0, 1.0j], [-1.0j, 0.0]])


def sigma_z():
    """Return the Pauli-Z operator."""
    return np.array([[1.0, 0.0j], [0.0j, -1.0]])


def sigma_plus():
    """Return the Pauli-creation operator."""
    return np.array([[0.0j, 1.0], [0.0, 0.0j]])


def sigma_minus():
    """Return the Pauli-annihilation operator."""
    return np.array([[0.0j, 0.0], [1.0, 0.0j]])


def identity_operator(dim: int):
    """Return the identity operator for the specified dimensions."""
    return np.eye(dim)


def dagger(Op: Array):
    """Return transpose conjugate of an operator."""
    return Op.T.conj()


@jit
def matrix_sqrt(Op: Array):
    """Returns matrix square root using jax based implementation.
    This works for any general matrix with positive eigenvalues.

    Uses jax.scipy.lingalg.sqrtm for the implementation.
    *NOTE - This function does not support automatic-differentiation.*
    """
    return sqrtm(Op)


@jit
def matrix_sqrt_psd(A):
    """Matrix square root of a Hermitian positive semi-definite matrix like a density matrix."""
    w, V = jnp.linalg.eigh(A)
    w_sqrt = jnp.sqrt(jnp.clip(w, min=0.0))  # clip tiny negatives from roundoff
    return (V * w_sqrt) @ dagger(V)


def partial_trace(rho: Array, dims: tuple[int, ...], keep: tuple[int, ...]):
    """Trace out all subsystems except those whose indices are in `keep`.

    Parameters
    ----------
    rho : Array
        Density matrix, shape (D, D) with D = prod(dims).
    dims : tuple[int, ...]
        Tuple of subsystem dimensions.
    keep : tuple[int, ...]
        Tuple of subsystem indices to keep. Counting starts from 0.

    Returns
    -------
    _type_
        _description_
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


def construct_annihilation_op(dim: int):
    """
    Create bosonic annihilation operator for a system with dimensions `dim`
    """
    return jnp.diag(jnp.sqrt(jnp.arange(1, dim, dtype=jnp.complex128)), k=1)


def construct_creation_op(dim: int):
    """
    Create bosonic creation operator for a system with dimensions `dim`
    """
    return jnp.diag(jnp.sqrt(jnp.arange(1, dim, dtype=jnp.complex128)), k=-1)


def construct_basis_state(dim: int, index: int) -> Array:
    """Generate pure basis state for a single system.

    Parameters
    ----------
    dim : int
        Dimension of the system.
    index : int
        Index of the state, for e.g., for fock state |0>, index = 0.
        Max index = dim - 1.

    Returns
    -------
    Array
        Basis state corresponding to the dimension and index.
    """
    if index >= dim:
        raise Exception(f"`index` has to be less than `dim`. Got dim={dim}, index={index}.")

    state = np.zeros(dim)
    state[index] = 1
    return state


def construct_composite_basis_state(dims: tuple[int, ...], index: tuple[int, ...]) -> Array:
    """Generate pure Basis state of a composite system based on the index.

    Parameters
    ----------
        dims : tuple[int, int]
            Subsystem dimensions as a tuple
        index : tuple[int, int]
            Index of the subsystem states as a tuple
    """

    if len(dims) != len(index):
        raise Exception("Length of dims and index must be same")

    if np.any((np.array(dims) - np.array(index)) <= 0):
        raise Exception("Index has to be less than subsystem dimension")

    individual_states = [construct_basis_state(dim, idx) for dim, idx in zip(dims, index)]
    return np.reshape(ntensor(individual_states), (-1, 1))


def convert_state_to_dm(state: Array):
    """Convert a pure state into a density matrix by taking the outer product."""
    return state @ state.T.conj()


@jit
def tensor(A: Array, B: Array) -> Array:
    """Tensor product of two operators OpA and OpB"""
    return jnp.kron(A, B)


def ntensor(Ops: list[Array]) -> Array:
    r"""Tensor product of a list of operators in the left to right order.

    Returns the operator:
    .. math::
        ntensor[A1, A2, ..., AN] = A1 \\otimes A2 \\otimes ... \\otimes AN.""
    """
    full_op = Ops[0]
    for op in Ops[1:]:
        full_op = tensor(full_op, op)
    return full_op


def tensor_product_with_identity(mat_list: list[Array], n: list[int], dims: list[int]) -> Array:
    r"""Put the matrices mat_list into a tensor product at positions `n`.

    All other positions are identity matrices:
    .. math::
        1 \\otimes \\dots \\otimes 1 \\otimes mat_list_1 \\otimes 1
            \\otimes \\dots \\otimes 1 \\otimes mat_list_2 \\dots
    The dimensions are assumed to be the same as the subsystems.

    Parameters
    ----------
    mat_list : List[Array]
        List of Matrices for tensor product
    n : list[int]
        List of indices for the each mat_list_i

    Returns
    -------
    Array
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


def construct_jump_operators_from_subsystems(subsystems: list[OpenSystem], dimensions: list[int]) -> list[Array]:
    """
    Gather jump operators from the subsystems and then tensor product them
    with identity to create the jump operators of the right dimension.

    Parameters
    ----------
    subsystems: list[OpenSystem]
        List of open systems in the same order as in CompositeSystem.
    dimensions: list[int]
        List of dimension of each subsystem.
    """
    all_collapse_ops = []
    for n, subsystem in enumerate(subsystems):
        jump_ops = subsystem.get_jump_operators()
        for jump_op in jump_ops:
            all_collapse_ops.append(tensor_product_with_identity([jump_op], [n], dimensions))
    return all_collapse_ops


## Helper functions for cross-package support

# Numpy


def np_func_to_jax_func(ham_func: Callable):
    """Convert a Numpy Hamiltonian function to JAX compatible function.

    Adds `vmap` capabilities to vectorize the computation over a batch of times (first variable).

    Parameters
    ----------
    ham_func : Callable
        Numpy based Hamiltonian function to convert to JAX and vmap compatible function.
    """

    def _jax_wrapper(times: Array, *args, **kwargs):
        one_time_func = lambda t: jnp.array(ham_func(t, *args, **kwargs))
        return vmap(one_time_func)(times)

    return _jax_wrapper


# QuTiP
def qobj_to_array(qobj):
    """Convert a QuTiP-JAX object to a JAX array.

    Parameters
    ----------
    qobj : Qobj
        QuTiP object.
    """
    return qobj.data._jxa


def qt_func_to_jax_func(ham_func: Callable):
    """Convert a QuTiP-JAX Hamiltonian function to JAX compatible function.

    Adds `vmap` capabilities to vectorize the computation over a batch of times (first variable).

    Parameters
    ----------
    ham_func : Callable
        QuTiP based Hamiltonian function to convert to JAX and vmap compatible function.
    """

    def _jax_wrapper(times: Array, *args, **kwargs):
        one_time_func = lambda t: ham_func(t, *args, **kwargs).data._jxa
        return vmap(one_time_func)(times)

    return _jax_wrapper
