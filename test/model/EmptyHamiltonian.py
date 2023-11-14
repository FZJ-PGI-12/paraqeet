import numpy as np

from cthree.model.Hamiltonian import Hamiltonian


class EmptyHamiltonian(Hamiltonian):
    """
    A Hamiltonian that is filled with zeros for all time steps.
    """

    dimension: int

    def __init__(self, dimension: int):
        super().__init__([], [], [], None)

        self.dimension = dimension

    def getMatrix(self, t: np.ndarray) -> np.ndarray:
        return np.zeros((len(t), self.dimension, self.dimension))
