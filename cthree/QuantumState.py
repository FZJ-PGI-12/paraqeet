import numpy as np


class QuantumState:
    """
    Represents a quantum state at a given time.
    """

    __vector: np.ndarray
    shape: tuple
    __time: np.ndarray

    def __init__(self, vec: np.ndarray, time: np.ndarray) -> None:
        self.__vector = vec
        self.shape = self.__vector.shape
        self.__time = time

    def getVector(self) -> np.ndarray:
        return self.__vector

    def getTime(self) -> np.ndarray:
        return self.__time
