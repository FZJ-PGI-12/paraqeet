from typing import List

from Optimisable import Optimisable
from Quantity import Quantity
from model import Hamiltonian


class Model(Optimisable):
    _hamiltonian: Hamiltonian

    def __construct(self, hamiltonian: Hamiltonian):
        self._hamiltonian = hamiltonian

    def getParameters(self) -> List[Quantity]:
        pass

    def getEquationOfMotion(self):
        pass
