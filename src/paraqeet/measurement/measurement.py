"""Abstract measurement interfaces whose values serve as optimization goal functions."""

from abc import ABC, abstractmethod
from typing import Protocol

import jax.numpy as jnp
from paraqeet.quantity import Array, Float


class Measurement(ABC):
    """Represents any observable and the process of measurement itself.

    The observable is measured after the propagation class
    has solved the equation of motion.
    """

    @abstractmethod
    def get_value(self, times: Array) -> Array | Float:
        """Measure the observable and return the value.

        Args:
            times: One-dimensional vector of timestamps.

        Returns:
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

        Args:
            times: One-dimensional vector of timestamps.

        Returns:
            Returns a Float if implemented by a subclass.
        """
        pass


class DifferentiableNormalizableMeasurement(Measurement):
    """Protocol for a class that is both NormalizableMeasurement and Differentiable.
    TODO: Write docstring
    Note:
        Only used for type-hinting, not to be used as a base class.
    """

    def __init__(self, meas: NormalizableMeasurement, propagation_gradient_func):
        self._measurement = meas
        self._propagation_gradient_func = propagation_gradient_func

    def get_value(self, times: Array) -> Array | Float:
        """Return the value of the measurement."""
        ...

    def get_gradient(self, times: Array) -> Array:
        """Compute the gradient.

        Args:
            times: One-dimensional vector of timestamps.

        Returns:
            The gradient of shape (n_params,).
        """
        states, dg_dp_list = self._propagation_gradient_func(times)
        final_state = states[-1]
        df_dp_list = []
        target_states = self._measurement._target_state
        f = self._measurement._overlap(final_state, target_states)
        for dg_dp in dg_dp_list[-1]:
            dfdp = self._fid_grad(f) * (self._overlap_grad(dg_dp, target_states).T @ dg_dp)
            df_dp_list.append(jnp.real(jnp.squeeze(dfdp)))
        return jnp.array(df_dp_list)  # (n_parameters,)

    def get_value_and_gradient(self, times: Array) -> tuple[Array | Float, Array]:
        """Return the value and the gradient of the measurement."""
        return self.get_value(times), self.get_gradient(times)

    def calculate_normalized_scalar(self, times: Array) -> Float:
        """Usually the same as get_value."""
        return self._measurement.calculate_normalized_scalar(times)
