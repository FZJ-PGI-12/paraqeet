from typing import List

from Optimisable import Optimisable
from Quantity import Quantity
from model import Hamiltonian


class Model(Optimisable):
    """
    Represents the equation of motion for a given Hamiltonian. Implementations can for example be the Schrödinger
    equation for a closed system, Lindbladian for an open system, or Hamilton's equations for a classical system.
    """
    _hamiltonian: Hamiltonian

    def __init__(self, hamiltonian: Hamiltonian):
        self._hamiltonian = hamiltonian

    def getParameters(self) -> List[Quantity]:
        pass

    def getEquationOfMotion(self):
        pass
