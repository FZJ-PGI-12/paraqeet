"""Class definition of a two level system (TLS) for testing."""

import numpy as np
from paraqeet.model.hamiltonian import Hamiltonian


class TLS(Hamiltonian):
    """A two-level system."""

    def __init__(self, drives=None):
        super().__init__(drives)
        self.sigma_p = np.array([[0j, 1], [0, 0]])
        self.dim = 2

    def get_parameters(self):
        return []

    def get_matrix_one_time(self, t):
        """Just sigma-X."""
        return self._drives[0].get_matrix_one_time(self.sigma_p, t)

    def gradient(self, t):
        """Gradient is just the drive matrix."""
        return self._drives[0].gradient(self.sigma_p, t)
