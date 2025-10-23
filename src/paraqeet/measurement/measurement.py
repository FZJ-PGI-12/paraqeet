"""Class definition of the Measurement model."""

from abc import ABC, abstractmethod

import jax.numpy as jnp

from paraqeet.quantity import Array


class Measurement(ABC):
    """Represents any observable and the process of measurement itself.

    The observable is measured after the propagation class
    has solved the equation of motion.

    Parameters
    ----------
    times: Array | None, optional
        One-dimensional vector of timestamps.

    """

    @abstractmethod
    def measure(self, times: Array) -> Array | float:
        """Measure the observable and returns the value.

        Parameters
        ----------
        times : Array
            One-dimensional vector of timestamps.
        projector : Array | None
            The projector matrix to restrict the operator.

        Returns
        -------
        Array or float
            This abstract method must return an Array or a float when
            implemented by subclasses. Might return multiple values.


        """
        pass


class NormalizableMeasurement(Measurement):
    """An abstract class for measurements providing normalized scalar value.

    Subclasses must implement the calculate_normalized_scalar() method which would
    return a measured value between 0 and 1.
    """

    @abstractmethod
    def calculate_normalized_scalar(self, times: Array) -> float:
        """Measure the normalized observable.

        Returns a single scalar value between 0 and 1.
        This function must be implemented by subclasses.
        Parameters
        ----------
        times : Array
            One-dimensional vector of timestamps.
        projection : Array | None
            The projector matrix to restrict the operator.

        Returns
        -------
        float
            Returns a float if implemented by a subclass.

        """
        pass
