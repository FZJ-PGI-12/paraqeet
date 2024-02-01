from typing import List

import numpy as np

from cthree.Quantity import Quantity
from cthree.model.Hamiltonian import Hamiltonian


class Transmon(Hamiltonian):
    """
    Hamiltonian of an anharmonic oscillator.
    """
    __dimension: int
    __frequency: Quantity
    __anharmonicity: Quantity
    __numOp: np.ndarray

    def __init__(self, dimension: int, frequency: Quantity, anharmonicity: Quantity):
        self.__dimension = dimension
        self.__frequency = frequency
        self.__anharmonicity = anharmonicity
        self.__numOp = np.diag(np.arange(0, self.__dimension))

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
        return [self.__frequency, self.__anharmonicity]

    def getMatrix(self, t: np.ndarray) -> np.ndarray:
        H = (self.__frequency.getValue() * self.__numOp +
             0.5 * self.__anharmonicity.getValue() * self.__numOp @ (self.__numOp - np.eye(self.__dimension)))
        return self._repeatInTime(H, t)

    def gradient(self, t: np.ndarray) -> List[np.ndarray]:
        # Derivatives wrt to the frequency and the anharmonicity
        gradFreq = self.__numOp
        gradAnharm = 0.5 * self.__numOp @ (self.__numOp - np.eye(self.__dimension))
        return [self._repeatInTime(gradFreq, t), self._repeatInTime(gradAnharm, t)]
