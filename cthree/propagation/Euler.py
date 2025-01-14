"""Class definition of the Euler propagation model."""

import numpy as np

from cthree.Quantity import Quantity
from cthree.model.Model import EquationOfMotion
from cthree.propagation.StatePropagation import StatePropagation


class Euler(StatePropagation):
    r"""Simple implementation of first order Euler propagation.

    Solves the equation of motion d/dt psi(t) = F(psi(t), t)
    with a finite step size d as psi(t+d) = psi(t) + F(psi(t), t).
    The step size can be variable and is calculated from the time array that is
    passed to the propagate function.

    Parameters
    ----------
    model : cthree.model.Model
        Represents the equation of motion for a given Hamiltonian.

    """

    def __init__(self, model: EquationOfMotion):
        super().__init__(model)

    def getParameters(self) -> list[Quantity]:
        """Get a list of parameters of the system.

        Returns
        -------
        List[Quantity]
            List of optimisable parameters of the system.

        """
        return []

    def propagate(self, time: np.ndarray) -> np.ndarray:
        """Calulate the first order Euler propagation.

        Performs the actual propagation calculation.

        Parameters
        ----------
        time : numpy.ndarray
            Vector of time samples.

        Returns
        -------
        numpy.ndarray
            Results of the Euler propagation.

        """
        equationsOfMotion = self._model.getMatrix(time)

        dt = time[1:] - time[0:-1]
        shape1 = (len(time),)
        shape2 = self._initialState.shape
        shape = shape1 + shape2
        states = np.zeros(shape=shape, dtype=np.complex128)
        states[0] = self._initialState
        for i in range(len(dt) - 1):
            states[i + 1] = states[i] + dt[i] * equationsOfMotion[i] @ states[i]

        return states
