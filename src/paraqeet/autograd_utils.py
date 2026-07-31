"""Helper functions for automatic differentiation."""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jax import jit, linearize, vjp, vmap

from paraqeet.quantity import Array


def get_value_and_jacobian_rev(
    f: Callable, argnums: int | tuple[int, ...] = 0
) -> Callable[..., tuple[Array, Array | tuple[Array, ...]]]:
    """Compute Jacobian of f w.r.t. specified arguments via vjp (reverse-mode AD).

    The function ``f`` has to be jax ``jit`` and ``grad`` compatible.
    Returns a ``get_value_and_gradient`` that returns the value and the gradients w.r.t. argnums.

    It supports functions with complex + vector valued inputs, and complex + vector valued outputs.
    Reverse mode is preferable when the number of inputs are larger than the outputs.

    Args:
        f: JAX-jit compatible function to be differentiated.
        argnums: int or tuple of ints (similar to jax.grad).
            Arguments for which the gradients are computed.

    Returns:
        A function that returns (value, gradient) for the given arguments.
        The gradient is a single Array if ``argnums`` is an int, and a tuple with one Array
        per entry of ``argnums`` if it is a tuple.
    """
    if isinstance(argnums, int):
        argnums_is_int = True
        indices: tuple[int, ...] = (argnums,)
    else:
        argnums_is_int = False
        indices = argnums

    @jit
    def value_and_jacobian_func(*args: Any, **kwargs: Any) -> tuple[Array, Array | tuple[Array, ...]]:
        diff_args = tuple(args[i] for i in indices)

        # Build a partial function that only depends on the args to differentiate with
        # The arguments not in diff_args are not traced and hence not differentiated with.
        def f_partial(*diff_args_: Any) -> Array:
            full_args_list = list(args)
            for i, a in zip(indices, diff_args_):
                full_args_list[i] = a
            result: Array = f(*full_args_list, **kwargs)
            return result

        y, vjp_fn = vjp(f_partial, *diff_args)

        # Scalar output
        if y.ndim == 0:
            grads = vjp_fn(jnp.ones_like(y))
            return y, grads[0] if argnums_is_int else grads

        # Array output: feed basis vectors covering the flattened output
        Identity = jnp.eye(y.size, dtype=y.dtype).reshape((y.size,) + y.shape)
        jac_tuple = vmap(vjp_fn)(Identity)

        reshaped = tuple(jac.reshape(y.shape + jnp.shape(diff_args[i])) for i, jac in enumerate(jac_tuple))

        return y, (reshaped[0] if argnums_is_int else reshaped)

    # `jit` returns JitWrapped object, fixing the signature here.
    wrapped: Callable[..., tuple[Array, Array | tuple[Array, ...]]] = value_and_jacobian_func
    return wrapped


def get_value_and_jacobian_fwd(
    f: Callable, argnums: int | tuple[int, ...] = 0
) -> Callable[..., tuple[Array, Array | tuple[Array, ...]]]:
    """Compute Jacobian of f w.r.t. specified arguments via jvp (forward-mode AD).

    The function ``f`` has to be jax ``jit`` and ``grad`` compatible.
    Returns a ``get_value_and_gradient`` that returns the value and the gradients w.r.t. argnums.

    It supports functions with complex + vector valued inputs, and complex + vector valued outputs.
    Forward mode is preferable when the number of inputs are smaller than the outputs.

    Args:
        f: JAX-jit compatible function to be differentiated.
        argnums: int or tuple of ints (similar to jax.grad).
            Arguments for which the gradients are computed.

    Returns:
        A function that returns (value, gradient) for the given arguments.
        The gradient is a single Array if ``argnums`` is an int, and a tuple with one Array
        per entry of ``argnums`` if it is a tuple.
    """
    if isinstance(argnums, int):
        argnums_is_int = True
        indices: tuple[int, ...] = (argnums,)
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

        y, jvp_fn = linearize(f_partial, *diff_args)

        jacobians = []
        for k, x in enumerate(diff_args):
            x_shape = jnp.shape(x)
            x_size = jnp.size(x)

            # Feed basis vectors covering the flattened input, with zero tangents
            # for the other differentiable arguments.
            Identity = jnp.eye(x_size, dtype=jnp.result_type(x)).reshape((x_size,) + x_shape)

            def push_tangent(v, k=k):
                tangents = tuple(v if j == k else jnp.zeros_like(a) for j, a in enumerate(diff_args))
                return jvp_fn(*tangents)

            jac = vmap(push_tangent)(Identity)
            # Move the input axis last to match the (y.shape + x.shape) convention
            # of the reverse-mode version.
            jacobians.append(jnp.moveaxis(jac, 0, -1).reshape(y.shape + x_shape))

        # ignoring mypy as the function is JitWrapped
        return y, (jacobians[0] if argnums_is_int else tuple(jacobians))  # type: ignore

    # `jit` returns JitWrapped object, fixing the signature here.
    wrapped: Callable[..., tuple[Array, Array | tuple[Array, ...]]] = value_and_jacobian_func
    return wrapped


def get_jacobian_func(f: Callable) -> Callable[..., Array]:
    """Return a function that computes the Jacobian of f w.r.t. its **first** arg via vjp.

    Note: This function has limited functionality. Use ``get_value_and_jacobian`` for wider scope.
    """

    @jit
    def jac_fn(x: Array, *args: Any, **kwargs: Any) -> Array:
        # Fix all the values except the first
        f_first = lambda x_: f(x_, *args, **kwargs)
        y, vjp_fn = vjp(f_first, x)

        # For scalar outputs
        if y.ndim == 0:
            scalar_grad: Array = vjp_fn(jnp.ones_like(y))[0]
            return scalar_grad

        # For Array outputs, seed the vjp with one basis covector per output element.
        seeds = jnp.eye(y.size, dtype=y.dtype).reshape((y.size,) + y.shape)
        jac_flat = vmap(vjp_fn)(seeds)[0]
        jac: Array = jac_flat.reshape(y.shape + x.shape)
        return jac

    # `jit` returns JitWrapped object, fixing the signature here.
    wrapped: Callable[..., Array] = jac_fn
    return wrapped
