import jax.numpy as jnp
import numpy as np
from jax import jit, vjp, vmap

from paraqeet.propagation.utils import convert_vec_to_dm
from paraqeet.quantity import Array


@jit
def overlap_state_vector(final_state: Array, target_state: Array, *args, **kwargs) -> Array:
    """Compute the overlap of state vectors."""
    return jnp.vdot(target_state, final_state)


@jit
def overlap_density_matrix(final_state: Array, target_state: Array, *args, **kwargs) -> Array:
    """Compute the overlap of density matrices."""
    # TODO: Make sure this works for mixed states: implement fidelity using sqrt(rho)
    return jnp.linalg.trace(jnp.matmul(target_state, final_state))


def overlap_vectorized_density_matrix(final_state: Array, target_state: Array, *args, **kwargs) -> Array:
    """Compute the overlap of density matrices."""
    # TODO: Make sure this works for mixed states: implement fidelity using sqrt(rho)
    dim = jnp.sqrt(target_state.shape[0]).astype(int)
    target_state = convert_vec_to_dm(target_state, dim)
    final_state = convert_vec_to_dm(final_state, dim)
    return jnp.linalg.trace(jnp.matmul(target_state, final_state))


def vjp_jacobian(f):
    """Returns a function that computes the Jacobian of f w.r.t. its first arg via vjp."""

    def jac_fn(x, *args, **kwargs):
        # Fix all the values except the first
        f_first = lambda x_: f(x_, *args, **kwargs)
        y, vjp_fn = vjp(f_first, x)

        # For scalar outputs
        if y.ndim == 0:
            return vjp_fn(jnp.ones_like(y))[0]

        # For Array outputs
        I = jnp.eye(y.size).reshape((y.size,) + y.shape)
        jac_flat = vmap(vjp_fn)(I)[0]
        return jac_flat.reshape(y.shape + x.shape)

    return jac_fn


def construct_times(time, ti, resolution):
    """Construct one-dimensional vector of time.

    Interpolate the user-specified times to match the propagation resolution.

    Parameters
    ----------
    time: Array
        Array of timesteps.
    ti: int
        Snapshot of the time at a current step

    Returns
    -------
    Array
        Array of timestamps in specified resolution.
    int
        Difference in time step.

    """
    t0 = time[ti - 1]
    t1 = time[ti]
    steps = int(np.ceil((t1 - t0) * resolution))
    times = jnp.linspace(t0, t1, steps, endpoint=False)
    if steps < 2:
        dt = t1 - t0
    else:
        dt = times[1] - times[0]
    return times, dt
