"""Utilies for model construction."""

import jax.numpy as jnp

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


def repeat(mat: Array, num: int) -> Array:
    """Repeats the matrix mat for each timestep in the times array.

    Returns an array with shape [t, n, m] where 't' is the
    number of time steps and 'mat' is an 'n' times 'm' matrix.

    Parameters
    ----------
    mat: Array
        Input matrix for repetition.
    num : int
        Number of times of repetition.

    Returns
    -------
    Array
        Repeated matrix for further computation.

    """
    return mat.reshape((1,) + mat.shape).repeat(num, axis=0)


# def get_collapseops(self) -> list[tuple[Array, Array]]:
#     """
#     Gather collapse operators from the subsystems and then tensor product them
#     with identity to create the collapse operators of the right dimension.
#     """
#     all_collapse_ops = []
#     for n, subsystem in enumerate(self._subsystems):
#         # TODO Could be solved better.
#         if isinstance(subsystem, OpenSystem):
#             rates_and_cols = subsystem.get_collapseops()
#             for rate, col_op in rates_and_cols:
#                 all_collapse_ops.append((rate, tensor_product_with_identity([col_op], [n], self._dimensions)))
#     return all_collapse_ops
