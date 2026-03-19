"""Class definition of the Euler propagation model."""

import jax.numpy as jnp

from paraqeet.exceptions import ConfigurationException
from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array, Quantity


class Euler(Propagation):
    r"""Simple implementation of first order Euler propagation.

    Solves the equation of motion d/dt psi(t) = F(psi(t), t)
    with a finite step size d as psi(t+d) = psi(t) + F(psi(t), t).
    The step size can be variable and is calculated from the time array that is
    passed to the propagate function.

    Parameters
    ----------
    model: Model
        Represents the equation of motion for a given Hamiltonian.

    """

    def __init__(self, model: EquationOfMotion, resolution: float = 1e9):
        # TODO: setting a default value for resolution
        super().__init__(model, resolution)

    def get_parameters(self) -> list[Quantity]:
        """Get a list of parameters of the system.

        Returns
        -------
        list[Quantity]
            List of optimizable parameters of the system.

        """
        return []

    def propagate(self, time: Array) -> Array:
        """Calulate the first order Euler propagation.

        Performs the actual propagation calculation.

        Parameters
        ----------
        time: Array
            Vector of time samples.

        Returns
        -------
        Array
            Results of the Euler propagation.

        """
        if len(time) < 2:
            raise ValueError("Euler.propagate needs at least two time points.")

        if self._initial_state is None:
            raise ConfigurationException("Initial state is not set")
        if self._eom_func is not None:
            equations_of_motion = self._eom_func.get_value(time)
        else:
            raise ConfigurationException("No equation of motion is configured.")

        dt = time[1:] - time[0:-1]
        states = [self._initial_state]
        for i in range(len(dt)):
            states.append(states[-1] + dt[i] * equations_of_motion[i] @ states[-1])

        return jnp.array(states)  # Jax arrays are immutable, so listing and then packing for return
