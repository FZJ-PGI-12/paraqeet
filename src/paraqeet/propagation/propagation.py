"""Abstract base classes for solvers that propagate the equation of motion and its gradient in time."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, override

import jax.numpy as jnp

from paraqeet.differentiable import Differentiable
from paraqeet.propagation.utils import construct_times
from paraqeet.quantity import Array


class Propagation(ABC):
    """Abstract base class for solvers of the equations of motion.

    The implemented propagation methods use vectorized operations to evaluate evolution of a
    single state, batch of states or entire propagator with the same code.
    For evaluating propagation of a propagator ``initial_state``
    can also be set to an initial `propagator`.

    Note:
        `Propagator` here refers to both unitary matrices for closed system and quantum channel for
        open quantum systems.
    """

    _eom_func: Callable[[Array], Array]
    _resolution: float
    _initial_state: Array

    def __init__(self, eom_func: Callable[[Array], Array], resolution: float, initial_state: Array) -> None:
        """
        Args:
            eom_func: A function that gives the equation of motion.
            resolution: Propagation resolution used to solve the equation of motion.
                The corresponding time step dt = 1/resolution
            initial_state: State at the beginning of the simulation.
        """
        self._eom_func = eom_func
        self._resolution = resolution
        self.initial_state = initial_state

    @property
    def resolution(self) -> float:
        """Return the propagation resolution."""
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: float) -> None:
        """Set the propagation resolution."""
        self._resolution = resolution

    @property
    def initial_state(self) -> Array:
        """Return initial state."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: Array) -> None:
        """Set initial state.

        The initial states can be thought of a collection of column vectors and
        all of them are propagated simultaneously by multiplying them with the `propagator`.
        So the initial states can in general would be a n x m matrix,
        where n is the system dimension and m are the number of initial states chosen.
        """
        # TODO: Provide explicit wrappers for multiple initial states or density vectors
        self._initial_state = jnp.array(state, dtype=jnp.complex128)

    @abstractmethod
    def _propagate(self, eom: Array, state: Array, steps: Array, *args: Any, **kwargs: Any) -> Array:
        """Propagate the state/propagator in time by solving the EOM.

        Note:
            It is recommended to make this function a 'pure' JAX function
            supporting JIT and gradient using vjp (reverse-mode AD).
            The type of arguments should be jax.Array.
            It is recommended to the user to add a `@jit` decorator with the
            appropriate `static_argnums`.
            For gradients using AD, derivative of this method is computed with
            respect to the first parameter (eom).

        Args:
            eom: Equation of motion for some time points as an Array.
            state: Initial state/propagator for propagation
            steps: Array of indices to iterate over.

        Returns:
            Propagated state/propagator by solving the EOM.
        """
        pass

    def get_value(self, times: Array) -> Array:
        """Return the solution of the equations of motion.

        Loop over all desired times in time at set resolution.

        The first dimension of the result will always be the time.
        Like in the model, the format of the other dimensions depends on the
        implementation and could for example be a propagated state vector or
        a propagator in matrix form.

        Args:
            times: Array of times.

        Returns:
            Array: Returns the solution of the equations of motion.

        Raises:
            ValueError: If fewer than two time points are given.

        """
        if len(times) < 2:
            raise ValueError("Propagation.get_value needs at least two time points.")

        psis = [self._initial_state]

        for ti in range(1, len(times)):
            step_times, dt = construct_times(times, ti, self._resolution)
            psis_t = psis[ti - 1]
            eom = self._eom_func(step_times + dt / 2) * dt
            psis_t = self._propagate(eom, psis_t, jnp.arange(0, len(step_times), 1))
            psis.append(psis_t)

        psis_arr = jnp.array(psis)
        return psis_arr


class DifferentiablePropagation(Differentiable):
    """Wraps a Propagation object to add gradient computation."""

    _prop: Propagation
    _eom_gradient_func: Callable[[Array], Array]

    def __init__(
        self,
        propagation: Propagation,
        eom_gradient_func: Callable[[Array], Array],
    ):
        self._prop = propagation
        self._eom_gradient_func = eom_gradient_func

    def _eom_and_gradient_func(self, times: Array) -> tuple[Array, Array]:
        """Return EOM and gradient of EOM for the given times array.

        Note:
            Default implementation computes the EOM and its gradient separately.
            *For cases where both the EOM and its gradient can be computed simultaneoulsy*
            *overwrite this method for decreasing the computational overhead.*

        Args:
            times: Array of time steps.
        """
        return self._prop._eom_func(times), self._eom_gradient_func(times)

    @override
    def get_value(self, times: Array) -> Array:
        """Return the solution of the equations of motion from the propagation method.

        Args:
            times: Array of times.

        """
        return self._prop.get_value(times)

    @override
    @abstractmethod
    def get_gradient(self, times: Array) -> Array:
        """Return gradient of propagated state/propagator w.r.t. specified parameters.

        Args:
            times: Array of time steps.
        """
        pass

    @override
    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
        """Return value and gradient of propagation.

        Args:
            times: Array of time points/
        """
        return self.get_value(times), self.get_gradient(times)
