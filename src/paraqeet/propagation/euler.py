"""Class definition of the Euler propagation model."""

import jax.numpy as jnp

from paraqeet.exceptions import ConfigurationException
from paraqeet.propagation.propagation import StatePropagation
from paraqeet.quantity import Array


class Euler(StatePropagation):
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

        eom_values = self._eom_func(time)
        dt = time[1:] - time[0:-1]
        states = [self._initial_state]
        for i in range(len(dt)):
            states.append(states[-1] + dt[i] * eom_values[i] @ states[-1])

        return jnp.array(states)  # Jax arrays are immutable, so listing and then packing for return
