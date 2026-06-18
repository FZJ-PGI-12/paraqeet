"""Class definition of the Measurement model."""

from abc import ABC, abstractmethod
from typing import Protocol

from paraqeet.differentiable import Differentiable
from paraqeet.quantity import Array, Float


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
    def measure(self, times: Array) -> Array | Float:
        """Measure the observable and returns the value.

        Parameters
        ----------
        times : Array
            One-dimensional vector of timestamps.
        projector : Array | None
            The projector matrix to restrict the operator.

        Returns
        -------
        Array or Float
            This abstract method must return an Array or a Float when
            implemented by subclasses. Might return multiple values.


        """
        pass


class NormalizableMeasurement(Measurement):
    """An abstract class for measurements providing normalized scalar value.

    Subclasses must implement the calculate_normalized_scalar() method which would
    return a measured value between 0 and 1.
    """

    @abstractmethod
    def calculate_normalized_scalar(self, times: Array) -> Float:
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
        Float
            Returns a Float if implemented by a subclass.

        """
        pass


class DifferentiableNormalizableMeasurement(Protocol):
    """Protocol for a class that is both NormalizableMeasurement and Differentiable"""

    def get_value(self, times: Array) -> Array | Float:
        """Returns the value of the measurement"""
        ...

    def get_gradient(self, times: Array) -> Array:
        """Returns the gradient of the measurement"""
        ...

    def get_value_and_gradient(self, times: Array) -> tuple[Array | Float, Array]:
        """Returns the value and the gradient of the measurement"""
        ...

    def measure(self, times: Array) -> Array | Float:
        """Usually the same as get_value"""
        ...

    def calculate_normalized_scalar(self, times: Array | Float) -> Float:
        """Usually the same as get_value"""
        ...
