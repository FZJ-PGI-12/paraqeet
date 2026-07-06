"""Class definition of the optimizable model."""

from collections.abc import Callable

from paraqeet.differentiable import Differentiable
from paraqeet.quantity import Array


class EquationOfMotion(Differentiable):
    """Represents the equation of motion of a system, assumed to be of the form

    d x /d t = A(t) x

    with x a vector or more generally a matrix characterizing the system, and A(t)
    another time-dependent matrix.

    Implementations can for example be the Schrödinger equation for a
    closed system, Lindbladian for an open system, or Hamilton's equations
    for a classical system. As these always involve the definition of a
    Hamiltonian function this needs to be passed together with a function
    that returns its gradient. The get_value and get_gradient methods
    should return the value and the gradient of A(t).
    """

    _hamiltonian_func: Callable[[Array], Array]
    _hamiltonian_gradient_func: Callable[[Array], Array]

    def __init__(
        self,
        hamiltonian_func: Callable[[Array], Array],
        hamiltonian_gradient_func: Callable[[Array], Array],
    ):
        """
        Args:
            hamiltonian_func: Function that returns the Hamiltonian at different
                times.
            hamiltonian_gradient_func: Function that returns the gradient of the
                Hamiltonian at different times.
        """
        self._hamiltonian_func = hamiltonian_func
        self._hamiltonian_gradient_func = hamiltonian_gradient_func
