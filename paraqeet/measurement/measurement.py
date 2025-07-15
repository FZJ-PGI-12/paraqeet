"""Class definition of the Measurement model."""

import jax.numpy as jnp

from paraqeet.optimisable import Optimisable
from paraqeet.quantity import Array


class Measurement(Optimisable):
    """Represents any observable and the process of measurement itself.

    The observable is measured after the propagation class
    has solved the equation of motion.

    Parameters
    ----------
    times: Array | None, optional
        One-dimensional vector of timestamps.

    """

    # Fields for tracing and projecting before the measurement
    __input_dimensions: list[int] | None = None
    __output_dimensions: list[int] | None = None
    __projector: Array | None = None
    _times: Array

    def __init__(self, times: Array):
        self._times = times

    def measure(self) -> Array | float:
        """Measure the observable and returns the value.

        Abstract Method. This function must be implemented by subclasses.

        Returns
        -------
        Array or float
            This abstract method must return a paraqeet.quantity.Array or a float when
            implemented by subclasses. Might return multiple values.

        Raises
        ------
        NotImplementedError
            If a subclass does not implement the measure method, raise an error.

        """
        return self.measure_normalised_scalar()

    def measure_scalar(self) -> float:
        """Measure the observable.

        Returns a scalar value. This function must be implemented by subclasses, unless identical to
        self.measure_normalised_scalar().
        """
        return self.measure_normalised_scalar()

    def measure_normalised_scalar(self) -> float:
        """Measure the normalised observable.

        Returns a single scalar value between 0 and 1, 1 representing the perfect result, required for use with most
        optimisations. This function must be implemented by subclasses.

        Returns
        -------
        Array
            Returns a paraqeet.quantity.Array if implemented by a subclass.

        """
        raise NotImplementedError()

    def measure_with_gradient(self) -> tuple[float, Array]:
        """Measure with gradient.

        Compute the measurement value as in measureNormalised()
        but with the gradient wrt to parameters.

        Returns
        -------
        Tuple[float, Array]
            Tuple of function value as bare float and gradient of shape (n_parameters,)

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
            if jnp.any(jnp.array(self.__input_dimensions) < 0) or jnp.any(jnp.array(self.__output_dimensions) < 0):
                raise RuntimeError("Dimensions must not be negative")
            if jnp.any(jnp.array(self.__input_dimensions) < jnp.array(self.__output_dimensions)):
                raise RuntimeError("Output dimensions can not be larger than input dimensions")
            if sum(output_dimensions) == 0:
                raise RuntimeError("All output dimensions can not be 0")

            P = jnp.eye(1)
            for dimIn, dimOut in zip(input_dimensions, output_dimensions):
                dim2 = dimOut if dimOut > 0 else 1
                P = jnp.kron(P, jnp.eye(dimIn, dim2, dtype=jnp.float64))
            self.__projector = P

    def _preprocess_matrix(self, operator: Array) -> Array:
        """Perform any preprocessing on the "operator" that was registered.

        Operator could be unitary matrices, density matrices.
        Subclasses should call this function before computing
        the measured value.

        Parameters
        ----------
        operator : Array
            Takes an array of Propagator/ density matrices as input.

        Returns
        -------
        Array
            The modified propagator.

        """
        if self.__projector is not None:
            operator = self.__projector.T @ operator @ self.__projector
        return operator

    def _preprocess_vector(self, states: Array) -> Array:
        """Perform any preprocessing on the "states" that were registered.

        States could be a single state or batch of state vectors.
        Subclasses should call this function before computing the
        measured value.

        Parameters
        ----------
        states : Array
            Single state or batch of state vectors.

        Returns
        -------
        Array
            The modified propagator.

        """
        if self.__projector is not None:
            if states.shape[-1] == 1:
                states = self.__projector.T @ states
            else:
                states = jnp.reshape(states, states.shape + (1,))
                states = self.__projector.T @ states
                states = jnp.squeeze(states, axis=-1)
        return states
