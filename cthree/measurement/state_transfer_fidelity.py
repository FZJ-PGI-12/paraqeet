"""The class definition of state transfer fidelity model."""

from collections.abc import Callable
from cthree.quantity import Quantity

from cthree.measurement.measurement import Measurement
from cthree.propagation.propagation import Propagation

import jax
import jax.numpy as jnp
from jax import Array, grad, jit
from jax.typing import ArrayLike

import warnings

jax.config.update("jax_enable_x64", True)


class StateTransferFidelity(Measurement):
    """Fidelity measure that compares overlap of the initial and final state.

    Parameters
    ----------
    propagation : cthree.measurement.Propagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    initial_state : jax.typing.ArrayLike
        Initial state.
    target_state : jax.typing.ArrayLike
        Target state.
    times : jax.typing.ArrayLike
        One-dimensional vector of timestamps.

    """

    _initial_state: ArrayLike
    _target_state: ArrayLike
    _propagation: Propagation

    def __init__(
        self,
        propagation: Propagation,
        initial_state: ArrayLike,
        target_state: ArrayLike,
        times: ArrayLike,
    ):
        super().__init__(times=times)
        self._propagation = propagation
        self._initial_state = initial_state
        self._target_state = target_state
        if target_state.shape != initial_state.shape:
            warnings.warn(
                UserWarning(
                    f"Different shapes for target_state({target_state.shape})"
                    f"and initial_state({initial_state.shape}) detected."
                    " Use restrict_subsystems to project states to "
                    "the same shape before measuring."
                )
            )
        self._propagation.set_initial_state(self._initial_state)

    @staticmethod
    def _fid(overlap):
        return jnp.abs(overlap) ** 2

    def measure(self) -> ArrayLike:
        """Measure overlap between initial and target state.

        Returns
        -------
        jax.typing.ArrayLike
            Overlap between initial and target state in a JAX ArrayLike format.

        """
        states = self._propagation.propagate(time=self._times)
        states = self._preprocess_vector(states)
        final_state = states[-1]
        f = jnp.vdot(self._target_state, final_state)
        return self._fid(f)

    def measure_with_gradient(self) -> tuple[Array, Array]:
        """Compute function value and corresponding gradient.

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        states, dg_dp_list = self._propagation.gradient(time=self._times)
        states = self._preprocess_vector(states)
        dg_dp_list = self._preprocess_vector(dg_dp_list)
        final_state = states[-1]
        dF_dp = []
        f = jnp.vdot(self._target_state, final_state)
        for dg_dp in dg_dp_list[-1]:
            g = jnp.vdot(self._target_state, dg_dp)
            dF_dp.append(jnp.real(f.conj() * g + f * g.conj()))  # chain rule for abs^2
        return self._fid(f), jnp.array(dF_dp)  # shape scalar, (n_parameters,)

    def get_parameters(self) -> list[Quantity]:
        """Get the parameters of the system.

        Returns
        -------
        list[cthree.quantity]
            List of parameters of the system.
        """
        return []


class StateTransferFidelityAD(StateTransferFidelity):
    """Fidelity measure that compares overlap of the initial and final state.

    Parameters
    ----------
    propagation : cthree.propagation.propagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    initial_state : jax.typing.ArrayLike
        Initial state.
    target_state : jax.typing.ArrayLike
        Target state.
    times : jax.typing.ArrayLike
        One-dimensional vector of timestamps.

    """

    __gradient_function: Callable | None

    def __init__(
        self,
        propagation: Propagation,
        initial_state: ArrayLike,
        target_state: ArrayLike,
        times: ArrayLike,
    ):
        super().__init__(propagation, initial_state, target_state, times)
        self.__gradient_function = None

    def measure_with_gradient(self) -> tuple[Array, Array]:
        """Measure with gradient.

        Overwrite inherited `measureWithGradient` to calculate
        gradients using AD.

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        if self.__gradient_function is None:
            self.__gradient_function = jit(grad(self._fid, argnums=0))

        states, dg_dp_list = self._propagation.gradient(time=self._times)
        states = self._preprocess_vector(states)
        dg_dp_list = self._preprocess_vector(dg_dp_list)
        final_state = states[-1]
        dF_dp = []
        f = jnp.vdot(self._target_state, final_state)
        for dg_dp in dg_dp_list[-1]:
            g = jnp.vdot(self._target_state, dg_dp)
            dfdp = self.__gradient_function(f) * g
            dF_dp.append(jnp.real(dfdp))
        return self._fid(f), jnp.array(dF_dp)  # shape scalar, (n_parameters,)


class StateTransferFidelityGRAPE(StateTransferFidelity):
    """Fidelity measure that compares overlap of the initial and final state.

    For GRAPE the optimisable parameters are vector quantities.

    Parameters
    ----------
    propagation : cthree.measurement.Propagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    initial_state : jax.typing.ArrayLike
        Initial state.
    target_state : jax.typing.ArrayLike
        Target state.
    times : jax.typing.ArrayLike
        One-dimensional vector of timestamps.

    """

    _propagation: Propagation

    def measure_with_gradient(self) -> tuple[Array, Array]:
        """Compute function value and corresponding gradient.

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        states, grads = self._propagation.gradient(time=self._times)
        final_state = states[-1]
        f = jnp.vdot(self._target_state, final_state)
        grads = 0.5 * jnp.real(f.conj() * grads + grads.conj() * f).flatten()
        return self._fid(f), grads  # shape scalar, (n_parameters,)


class StateTransferFidelityOpenGRAPE(StateTransferFidelity):
    """Fidelity measure that compares overlap of the initial and final state.

    For GRAPE the optimisable parameters are vector quantities.

    Parameters
    ----------
    propagation : cthree.measurement.Propagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    initial_state : jax.typing.ArrayLike
        Initial state.
    target_state : jax.typing.ArrayLike
        Target state.
    times : jax.typing.ArrayLike
        One-dimensional vector of timestamps.

    """

    _propagation: Propagation

    def measure(self):
        """Measure overlap between initial and target density matrices.

        Returns
        -------
        jax.typing.ArrayLike
            Overlap between initial and target state in a JAX ArrayLike format.

        """
        states = self._propagation.propagate(time=self._times)
        states = self._preprocess_vector(states)
        final_state = states[-1]
        f = jnp.linalg.trace(jnp.matmul(self._target_state, final_state))
        return jnp.real(f)

    def measure_with_gradient(self) -> tuple[Array, Array]:
        """Compute function value and corresponding gradient.

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            Tuple of function value and gradient of shape (n_parameters,).

        """
        states, grads = self._propagation.gradient(time=self._times)
        final_state = states[-1]
        f = jnp.linalg.trace(jnp.matmul(self._target_state, final_state))
        gradients = jnp.linalg.trace(grads)
        return jnp.real(f), jnp.real(gradients).flatten()  # shape scalar, (n_parameters,)
