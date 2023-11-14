from typing import List, Optional

import numpy as np

from cthree.signal.Generator import Generator


class Hamiltonian:
    """
    Matrix representation of a Hamiltonian.
    Contains subsystems, couplings, and drive lines.
    Takes care of frame transformations.
    """

    _subsystems: List
    _couplings: List
    _drives: List
    _generator: Generator

    def __init__(
        self,
        subsystems: List,
        couplings: List = [],
        drives: List = [],
        generator: Optional[Generator] = None,
    ):
        self._subsystems = subsystems
        self._couplings = couplings
        self._drives = drives
        self._generator = generator

    def getMatrix(self, t: np.ndarray) -> np.ndarray:
        """
        Return the matrix representation of the Hamiltonian.

        Args:
            t (np.ndarray): Vector of time samples

        Returns:
            np.ndarray: Hamiltonian of shape [t, n, n]  with t: time, n: hilbert space
        """
        sig = self._generator.generateSignal(t)
        return self._subsystems[0] + sig * self._drives[0]
