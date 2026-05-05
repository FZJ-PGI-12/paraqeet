"""Utilies for model construction."""

import jax.numpy as jnp

from paraqeet.model.system import OpenSystem
from paraqeet.quantity import Array


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
