"""Class definition for the Runge-Kutta Scipy propagation model."""

import numpy as np  # Using regular numpy for scipy interface
from scipy.integrate import RK45  # TODO: Replace with jax? Is there one?

from paraqeet.exceptions import ConfigurationException
from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.propagation.state_propagation import StatePropagation
from paraqeet.quantity import Quantity, Array


class RungeKutta(StatePropagation):
    """Propagation via the Runge-Kutta Scipy implementation.

    Uses scipy's Runge-Kutta implementation for propagating
    a state vector or density matrix.

    Parameters
    ----------
    model : Model
        Represents the equation of motion for a given Hamiltonian.
    initial_time_step : float | None, optional
        The initial time step for the adaptive time steps in RK45.

    """

    __initial_time_step: float | None

    def __init__(self, model: EquationOfMotion, initial_time_step: float | None = None):
        super().__init__(model)
        self._initial_state: Array
        self.__initial_time_step = initial_time_step

    def get_parameters(self) -> list[Quantity]:
        """Get a list of parameters of the system.

        Returns
        -------
        List[Quantity]
            List of optimisable parameters of the system.

        """
        return []

    def set_initial_state(self, state: Array):
        """Set the initial state for the propagation.

        Subclasses can access the state in the _initial_state field.

        Parameters
        ----------
        state : Array
            Parameter value to be set as the initial state for the propagation.

        """
        self._initial_state = np.reshape(state, (-1,))

    def propagate(self, time: Array) -> Array:
        """Return the solution of the equations of motion.

        Parameters
        ----------
        time : Array
            Any one-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the solution of the equations of motion.

        Raises
        ------
        ConfigurationException
            If the initial state is not set.
        ValueError
            If the propagation needs at least two time steps.

        """
        if self._initial_state is None:
            raise ConfigurationException("Initial state is not set")

        if len(time) < 2:
            raise ValueError("Runge-Kutta propagation needs at least two time steps")

        def callback(time, state):
            column_state = np.reshape(state, (-1, 1))
            return np.reshape(
                self._model.get_right_hand_side(np.array([time]), column_state),
                (-1,),
            )

        # Since RK45 uses adaptive time steps and does not guarantee
        # to return a state for each time stamp, this
        # function has to iterate over the time steps itself.
        states = [self._initial_state]
        for ti in range(1, len(time)):
            dt = self.__initial_time_step
            if dt is None or dt > time[ti] - time[ti - 1]:
                dt = float(time[ti] - time[ti - 1]) / 5

            # This is the scipy implementation of RK45, which is compatible with (non-jax) numpy
            integrator = RK45(
                fun=callback,
                t0=time[ti - 1],
                y0=np.reshape(states[-1], (-1,)),
                t_bound=time[ti],
                first_step=dt,
                vectorized=False,
            )

            while integrator.status == "running":
                integrator.step()
            states.append(integrator.y)
        return np.array(states)
