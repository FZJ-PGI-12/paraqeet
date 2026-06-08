"""Class definition of the Propagation model."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import jax.numpy as jnp

from paraqeet.differentiable import Differentiable
from paraqeet.quantity import Array


class Propagation(ABC):
    """Abstract base class for any implementation of the equations of motion.

    The right-hand side of the equation is provided by the underlying model.

    Parameters
    ----------
    model: Model
        Represents the equation of motion for a given Hamiltonian.
    resolution: float
        Propagation resolution used to solve the equation of motion.
        The corresponding time step dt = 1/resolution
    """

    _eom_func: Callable[[Array], Array]
    _resolution: float

    def __init__(self, eom_func: Callable[[Array], Array], resolution: float):
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
    def propagate(self, time: Array) -> Array:
        """Return the solution of the equations of motion.

        The first dimension of the result will always be the time.
        Like in the model, the format of the other dimensions depends on the
        implementation and could for example be a propagated state vector or
        a propagator in matrix form.

        Parameters
        ----------
        time: Array
            Any one-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the solution of the equations of motion.


        """
        # TODO: Distinguish between internal time (class property), i.e. the time grid of the
        # method vs. time points (input parameter) desired by other classes, e.g. Measurements
        pass


class StatePropagation(Propagation):
    """Abstract class to implement methods to propagate states."""

    _initial_state: Array

    def __init__(self, eom_func: Callable[[Array], Array], resolution: float, initial_state: Array):
        super().__init__(eom_func, resolution)
        self.initial_state = initial_state

    @property
    def initial_state(self):
        """Return initial state."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: Array):
        """Set initial state."""
        # TODO: Provide explicit wrappers for multiple initial states or density vectors
        self._initial_state = jnp.array(state, dtype=jnp.complex128)


# TODO: Does a differantiable propagation make sense physically? Should we rather have
# differantiable models and use composition instead?
# class DifferentiablePropagation(Propagation, Differentiable):
#     """Propagation methods that provide a get_value_and_gradient method.
#
#     Parameters
#     ----------
#     model: Model
#         Represents the equation of motion for a given Hamiltonian.
#     """
#
#     _resolution: float
#
#     @abstractmethod
#     def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
#         """Gradient method to be implemented
#
#         Parameters
#         ----------
#         times: Array
#             Array of timesteps.
#
#         Returns
#         -------
#         tuple[Array, Array]
#             First dimension is time, second dimension is the parameter.
#
#         """
#         pass
