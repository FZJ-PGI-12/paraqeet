"""Abstract base class for solvers that propagate the equation of motion in time."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import jax.numpy as jnp

from paraqeet.quantity import Array


class Propagation(ABC):
    """Abstract base class for solver of the equations of motion."""

    _eom_func: Callable[[Array], Array]
    _resolution: float

    def __init__(self, eom_func: Callable[[Array], Array], resolution: float) -> None:
        """
        Args:
            eom_func: A function that gives the equation of motion.
            resolution: Propagation resolution used to solve the equation of motion.
                The corresponding time step dt = 1/resolution
        """
        self._eom_func = eom_func
        self._resolution = resolution

    @property
    def resolution(self) -> float:
        """Return the propagation resolution."""
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: float) -> None:
        """Set the propagation resolution."""
        self._resolution = resolution

    @abstractmethod
    def get_value(self, times: Array) -> Array:
        """Return the solution of the equations of motion.

        The first dimension of the result will always be the time.
        Like in the model, the format of the other dimensions depends on the
        implementation and could for example be a propagated state vector or
        a propagator in matrix form.

        Args:
            times: Array of times.

        Returns:
            Array: Returns the solution of the equations of motion.


        """
        # TODO: Distinguish between internal time (class property), i.e. the time grid of the
        # method vs. time points (input parameter) desired by other classes, e.g. Measurements
        pass


class StatePropagation(Propagation):
    """Abstract class to implement methods to propagate states."""

    _initial_state: Array

    def __init__(self, eom_func: Callable[[Array], Array], resolution: float, initial_state: Array) -> None:
        """
        Args:
            eom_func: A function that gives the equation of motion.
            resolution: Propagation resolution used to solve the equation of motion.
                The corresponding time step dt = 1/resolution.
            initial_state: State at the beginning of the simulation.
        """
        super().__init__(eom_func, resolution)
        self.initial_state = initial_state

    @property
    def initial_state(self):
        """Return initial state."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: Array) -> None:
        """Set initial state."""
        # TODO: Provide explicit wrappers for multiple initial states or density vectors
        self._initial_state = jnp.array(state, dtype=jnp.complex128)
