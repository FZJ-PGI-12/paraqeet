from abc import abstractmethod
from typing import List, Tuple

import numpy as np

from cthree.Optimisable import Optimisable


class Measurement(Optimisable):
    """
    Represents any observable and the process of measurement itself. The observable is measured after the propagation
    class has solved the equation of motion.
    """

    # Fields for tracing and projecting before the measurement
    __inputDimensions: List[int] | None = None
    __outputDimensions: List[int] | None = None
    __projector: np.ndarray | None = None

    def __init__(self, times: np.ndarray | None = None):
        self._times = times

    @abstractmethod
    def measure(self) -> np.ndarray:
        """
        Measures the observable and returns the value. This function must be implemented by subclasses.
        """
        raise NotImplementedError()

    def measureNormalised(self) -> np.ndarray:
        """
        Measures the observable and returns the value between 0 and 1, 1 representing the perfect result.
        This function must be implemented by subclasses, unless identical to self.measure().
        """
        return self.measure()

    def measureWithGradient(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the measurement value as in measureNormalised() but with the gradient wrt to parameters.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of function value and gradient of shape (n_parameters,)

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()

    def restrictSubsystems(
        self, inputDimensions: List[int], outputDimensions: List[int] | None = None
    ) -> None:
        """
        Notifies the measurement class that the computed propagator should be projected to a subspace before doing the
        measurement. Dimensions of the subspaces are specified per subsystem.

        Parameters
        ----------
        inputDimensions: Actual dimensions of all subsystems
        outputDimensions: Desired dimensions of all subsystems. Individual values can be 0 to fully remove subsystems
                          from the propagator. The list can be None to disable projection.
        """
        self.__inputDimensions = inputDimensions
        self.__outputDimensions = outputDimensions
        self.__projector = None

        # Construct the projector matrix
        if outputDimensions is not None:
            if len(inputDimensions) != len(outputDimensions):
                raise RuntimeError(
                    "The input and output dimensions must contain the same number of subsystems"
                )
            if np.any(np.array(self.__inputDimensions) < 0) or np.any(
                np.array(self.__outputDimensions) < 0
            ):
                raise RuntimeError("Dimensions must not be negative")
            if np.any(
                np.array(self.__inputDimensions) < np.array(self.__outputDimensions)
            ):
                raise RuntimeError(
                    "Output dimensions can not be larger than input dimensions"
                )
            if np.sum(outputDimensions) == 0:
                raise RuntimeError("All output dimensions can not be 0")

            P = np.eye(1)
            for dimIn, dimOut in zip(inputDimensions, outputDimensions):
                dim2 = dimOut if dimOut > 0 else 1
                P = np.kron(P, np.eye(dimIn, dim2))
            self.__projector = P

    def _preprocess(self, operator: np.ndarray) -> np.ndarray:
        """
        Performs any preprocessing on the "operator" that was registered.
        Operator could be unitary matrices, density matrices or a single or batch of state vectors.
        Subclasses should call this function before computing the measured value.

        Parameters
        ----------
        operator: the Propagator/ States

        Returns
        -------
        The modified propagator
        """
        if self.__projector is not None:
            if (
                operator.shape[-1] == operator.shape[-2]
            ):  # For Unitary operator or Matrices
                operator = self.__projector.T @ operator @ self.__projector
            else:  # For states
                operator = self.__projector.T @ operator
        return operator
