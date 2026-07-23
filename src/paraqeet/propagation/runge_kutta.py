"""Class definition for the Runge-Kutta Scipy propagation model."""

from collections.abc import Callable
from typing import override

import numpy as np  # Using regular numpy for scipy interface
from scipy.integrate import RK45  # TODO: Replace with jax? Is there one?

from paraqeet.exceptions import ConfigurationException
from paraqeet.propagation.propagation import StatePropagation
from paraqeet.quantity import Array


class RungeKutta(StatePropagation):
    """Propagation via the Runge-Kutta Scipy implementation.

    Uses scipy's Runge-Kutta implementation for propagating
    a state vector or density matrix.
    """

    _initial_time_step: float

    def __init__(self, eom_func: Callable[[Array], Array], resolution: float, initial_state: Array) -> None:
        """
        Args:
            eom_func: A function that gives the equation of motion.
            resolution: Propagation resolution used to solve the equation of motion.
                The corresponding time step dt = 1/resolution.
            initial_state: State at the beginning of the simulation.
        """
        super().__init__(eom_func, resolution, initial_state)
        self._initial_time_step = 1 / resolution

    def set_initial_state(self, state: Array) -> None:
        """Set the initial state for the propagation.

        Subclasses can access the state in the _initial_state field.

        Args:
            state: Parameter value to be set as the initial state for the propagation.

        """
        self._initial_state = np.reshape(state, (-1,))

    @override
    def get_value(self, times: Array) -> Array:
        """Return the solution of the equations of motion.

        Args:
            times: Array of times.

        Returns:
            The solution of the equations of motion.

        Raises
            ConfigurationException: If the initial state is not set.
            ValueError: If the propagation needs at least two time steps.
        """
        if self._initial_state is None:
            raise ConfigurationException("Initial state is not set")

        if len(times) < 2:
            raise ValueError("RungeKutta.get_value needs at least two time steps")

        def callback(time: float, state: Array) -> Array:
            column_state = np.reshape(state, (-1, 1))
            return np.reshape(
                self._eom_func(np.array([time])) @ column_state,
                (-1),
            )

        # Since RK45 uses adaptive time steps and does not guarantee
        # to return a state for each time stamp, this
        # function has to iterate over the time steps itself.
        states = [self._initial_state]
        for ti in range(1, len(times)):
            dt = self._initial_time_step
            if dt is None or dt > times[ti] - times[ti - 1]:
                dt = float(times[ti] - times[ti - 1]) / 5

            # This is the scipy implementation of RK45, which is compatible with (non-jax) numpy
            integrator = RK45(
                fun=callback,
                t0=times[ti - 1],
                y0=np.reshape(states[-1], (-1,)),
                t_bound=times[ti],
                first_step=dt,
                vectorized=False,
            )

            while integrator.status == "running":
                integrator.step()
            states.append(np.reshape(integrator.y, (-1, 1)))
        return np.array(states)
