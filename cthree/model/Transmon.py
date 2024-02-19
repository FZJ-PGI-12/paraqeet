from typing import List

import numpy as np

from cthree.Quantity import Quantity
from cthree.model.Drive import Drive
from cthree.model.Hamiltonian import Hamiltonian


class Transmon(Hamiltonian):
    """
    Hamiltonian of an anharmonic oscillator. Optimisable parameters are the ground frequency and the anharmonicity.
    """
    __dimension: int
    __frequency: Quantity
    __anharmonicity: Quantity
    __annihilationOp: np.ndarray
    __numOp: np.ndarray

    def __init__(self, dimension: int, frequency: Quantity, anharmonicity: Quantity, drives: List[Drive] = None):
        super().__init__(drives=drives)
        self.__dimension = dimension
        self.__frequency = frequency
        self.__anharmonicity = anharmonicity
        self.__annihilationOp = np.sqrt(np.diag(np.arange(1, dimension, dtype=np.float64), k=1))
        self.__numOp = self.__annihilationOp.T @ self.__annihilationOp

    def dimension(self) -> int:
        return self.__dimension

    def getFrequency(self) -> Quantity:
        return self.__frequency

    def setFrequency(self, frequency: Quantity) -> None:
        self.__frequency = frequency

    def getAnharmonicity(self) -> Quantity:
        return self.__anharmonicity

    def setAnharmonicity(self, anharmonicity: Quantity) -> None:
        self.__anharmonicity = anharmonicity

    def getParameters(self) -> List[Quantity]:
        return [self.__frequency, self.__anharmonicity] + self._getDriveParameters()

    def getMatrix(self, t: np.ndarray) -> np.ndarray:
        H = (self.__frequency.getValue() * self.__numOp +
             0.5 * self.__anharmonicity.getValue() * self.__numOp @ (self.__numOp - np.eye(self.__dimension)))
        return self._repeat(H, t.shape[0]) + self._getDriveMatrix(self.__annihilationOp, t)

    def gradient(self, t: np.ndarray) -> List[np.ndarray]:
        # Derivatives wrt to the frequency and the anharmonicity
        gradFreq = self.__numOp
        gradAnharm = 0.5 * self.__numOp @ (self.__numOp - np.eye(self.__dimension))
        driveGradients = self._getDriveGradients(self.__annihilationOp, t)
        return [self._repeat(gradFreq, t.shape[0]), self._repeat(gradAnharm, t.shape[0])] + driveGradients
