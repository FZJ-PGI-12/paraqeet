"""Helper functions for automatic differentation."""

from collections.abc import Callable

import jax.numpy as jnp
from jax import jit, vjp, vmap

from paraqeet.quantity import Array


def get_value_and_jacobian(f: Callable, argnums: int | tuple[int] = 0) -> Callable:
    """Compute Jacobian of f w.r.t. specified arguments via vjp (reverse-mode AD).

    The function `f` has be jax `jit` and `grad` compatible.
    Returns a `get_value_and_gradient` that returns the value and the gradients w.r.t. argnums.

    It supports functions with complex + vector valued inputs, and complex + vector valued outputs.

    Parameters
    ----------
    f : Callable
        JAX-jit compatible function to be differentiated.
    argnums : int, tuple[int]
        int or tuple of ints (similar to jax.grad).
        Arguments for which the gradients are computed.

    Returns
    -------
    _type_
        _description_
    """
    if isinstance(argnums, int):
        argnums_is_int = True
        indices: tuple[int] = (argnums,)
    else:
        argnums_is_int = False
        indices = argnums

    @jit
    def value_and_jacobian_func(*args, **kwargs) -> tuple[Array, Array]:
        diff_args = tuple(args[i] for i in indices)

        # Build a partial function that only depends on the args to differentiate with
        # The arguments not in diff_args are not traced and hence not differentiated with.
        def f_partial(*diff_args_):
            full_args_list = list(args)
            for i, a in zip(indices, diff_args_):
                full_args_list[i] = a
            return f(*full_args_list, **kwargs)

        y, vjp_fn = vjp(f_partial, *diff_args)

        # Scalar output
        if y.ndim == 0:
            grads = vjp_fn(jnp.ones_like(y))
            return y, grads[0] if argnums_is_int else grads

        # Array output: feed basis vectors covering the flattened output
        Identity = jnp.eye(y.size, dtype=y.dtype).reshape((y.size,) + y.shape)
        jac_tuple = vmap(vjp_fn)(Identity)

        reshaped = tuple(jac.reshape(y.shape + jnp.shape(diff_args[i])) for i, jac in enumerate(jac_tuple))

        # ignoring mypy as the function is JitWrapped
        return y, (reshaped[0] if argnums_is_int else reshaped)  # type: ignore

    # ignoring mypy as the function is JitWrapped
    return value_and_jacobian_func  # type: ignore
