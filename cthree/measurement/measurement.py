"""Class definition of the Measurement model."""

from abc import abstractmethod

import numpy as np

from cthree.optimisable import Optimisable


class Measurement(Optimisable):
    """Represents any observable and the process of measurement itself.

    The observable is measured after the propagation class
    has solved the equation of motion.

    Parameters
    ----------
    times: numpy.ndarray | None, optional
        One-dimensional vector of timestamps.

    """

    # Fields for tracing and projecting before the measurement
    __input_dimensions: list[int] | None = None
    __output_dimensions: list[int] | None = None
    __projector: np.ndarray | None = None

    def __init__(self, times: np.ndarray | None = None):
        self._times = times

    @abstractmethod
    def measure(self) -> np.ndarray:
        """Measure the observable and returns the value.

        Abstract Method. This function must be implemented by subclasses.

        Returns
        -------
        numpy.ndarray
            This abstract method must return a Numpy ndarray when
            implemented by subclasses.

        Raises
        ------
        NotImplementedError
            If a subclass does not implement the measure method, raise an error.

        """
        raise NotImplementedError()

    def measure_normalised(self) -> np.ndarray:
        """Measure the normalised observable.

        Returns the value between 0 and 1, 1 representing the perfect result.
        This function must be implemented by subclasses,
        unless identical to self.measure().

        Returns
        -------
        numpy.ndarray
            Returns a Numpy ndarray if implemented by a subclass.

        """
        return self.measure()

    def measure_with_gradient(self) -> tuple[np.ndarray, np.ndarray]:
        """Measure with gradient.

        Compute the measurement value as in measureNormalised()
        but with the gradient wrt to parameters.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            Tuple of function value and gradient of shape (n_parameters,)

        Raises
        ------
        NotImplementedError
            If a subclass does not implement the measureWithGradient method,
            raise an error.

        """
        raise NotImplementedError()

    def restrict_subsystems(
        self,
        input_dimensions: list[int],
        output_dimensions: list[int] | None = None,
    ) -> None:
        """Restrict subsystem by projecting to a subspace.

        Notifies the measurement class that the computed propagator should be
        projected to a subspace before doing the measurement.
        Dimensions of the subspaces are specified per subsystem.

        Parameters
        ----------
        input_dimensions : List[int]
            Actual dimensions of all subsystems.
        output_dimensions : List[int] | None, optional
            Desired dimensions of all subsystems.
            Individual values can be 0 to fully remove subsystems
            from the propagator. The list can be None to disable projection.

        Raises
        ------
        RuntimeError
            If the input and output dimensions don't have the same
            number of subsystems.
        RuntimeError
            If the dimensions are negative.
        RuntimeError
            If the output dimensions are larger than the input dimensions.
        RuntimeError
            If all output dimensions are zero.

        """
        self.__input_dimensions = input_dimensions
        self.__output_dimensions = output_dimensions
        self.__projector = None

        # Construct the projector matrix
        if output_dimensions is not None:
            if len(input_dimensions) != len(output_dimensions):
                raise RuntimeError(
                    "The input and output dimensions must \
                        contain the same number of subsystems"
                )
            if np.any(np.array(self.__input_dimensions) < 0) or np.any(np.array(self.__output_dimensions) < 0):
                raise RuntimeError("Dimensions must not be negative")
            if np.any(np.array(self.__input_dimensions) < np.array(self.__output_dimensions)):
                raise RuntimeError("Output dimensions can not be larger than input dimensions")
            if np.sum(output_dimensions) == 0:
                raise RuntimeError("All output dimensions can not be 0")

            P = np.eye(1)
            for dimIn, dimOut in zip(input_dimensions, output_dimensions):
                dim2 = dimOut if dimOut > 0 else 1
                P = np.kron(P, np.eye(dimIn, dim2))
            self.__projector = P

    def _preprocess_matrix(self, operator: np.ndarray) -> np.ndarray:
        """Perform any preprocessing on the "operator" that was registered.

        Operator could be unitary matrices, density matrices.
        Subclasses should call this function before computing
        the measured value.

        Parameters
        ----------
        operator : numpy.ndarray
            Takes an array of Propagator/ density matrices as input.

        Returns
        -------
        numpy.ndarray
            The modified propagator.

        """
        if self.__projector is not None:
            operator = self.__projector.T @ operator @ self.__projector
        return operator

    def _preprocess_vector(self, states: np.ndarray) -> np.ndarray:
        """Perform any preprocessing on the "states" that were registered.

        States could be a single state or batch of state vectors.
        Subclasses should call this function before computing the
        measured value.

        Parameters
        ----------
        states : numpy.ndarray
            Single state or batch of state vectors.

        Returns
        -------
        numpy.ndarray
            The modified propagator.

        """
        if self.__projector is not None:
            if states.shape[-1] == 1:
                states = self.__projector.T @ states
            else:
                states = np.reshape(states, states.shape + (1,))
                states = self.__projector.T @ states
                states = np.squeeze(states, axis=-1)
        return states
