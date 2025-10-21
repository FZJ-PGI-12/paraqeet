"""Class definition of the Weighted Sum Goal model."""

import itertools

import jax.numpy as jnp
import numpy as np

from paraqeet.differentiable import Differentiable
from paraqeet.exceptions import ConfigurationException
from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.quantity import Array


class WeightedSumGoal(NormalizableMeasurement, Differentiable):
    """Combine multiple measurements into a single goal function.

    Parameters
    ----------
    measurements : list[Measurement]
        List of measurements.
    weights : Array
        List of weights.
    sum_of_squares_options : dict | None
        A dictionary that contains information about how to include the
        sum of square differences in the cost function. If not None
        the it must contain the following keys:
        weight : float
            The weight of the sum of square differences
        meas_bool : list[bool]
            A list of boolean of the same length as measurements. If
            an element is True then the corresponding measurement is included
            in the sum of square difference the goal function.
    measurement_in_sum_of_squares : list[Measurement] | None
        The list of measurements included in the sum of square difference cost
        function. It is None if sum_of_squares_options is None


    Raises
    ------
    ConfigurationException
        If number of measurements and weights are incompatible.
    UserWarning
        If the given weights are not normalized.

    """

    __measurements: list[NormalizableMeasurement]
    __weights: Array
    __sum_of_squares_options: dict | None
    __measurements_in_sum_of_squares: list[NormalizableMeasurement]

    def __init__(
        self, measurements: list[NormalizableMeasurement], weights: Array, sum_of_squares_options: dict | None = None
    ):
        super().__init__(jnp.array(0.0))
        self.__measurements = measurements
        self.__weights = weights
        self.__sum_of_squares_options = sum_of_squares_options
        if len(measurements) != len(weights):
            raise ConfigurationException(
                f"Incompatible number of measurements {len(measurements)} and weights {{len(weights)}}"
            )
        if sum_of_squares_options is not None:
            expected_keys = ["weight", "meas_bool"]
            provided_keys = list(sum_of_squares_options.keys())

            for key in provided_keys:
                if key not in expected_keys:
                    raise KeyError("The dictionary must have weight and meas_bool as keys")

            if not np.isclose(sum(weights) + sum_of_squares_options["weight"], 1.0):
                raise UserWarning("Total supplied weights are not normalized.")

            self.__measurements_in_sum_of_squares = [
                meas for meas, flag in zip(measurements, sum_of_squares_options["meas_bool"]) if flag
            ]
        else:
            if not np.isclose(sum(weights), 1.0):
                raise UserWarning("Supplied weights are not normalized.")
            self.__measurements_in_sum_of_squares = []

    @property
    def measurements(self) -> list[NormalizableMeasurement]:
        """Returns the list of measurement"""
        return self.__measurements

    @property
    def weights(self) -> Array:
        """Returns the weights used in the weighted goal function"""
        return self.__weights

    @property
    def sum_of_square_options(self) -> dict | None:
        """Returns the dictionary with the options about the sum of square differences goal function"""
        return self.__sum_of_squares_options

    @property
    def measurements_in_sum_of_squares(self) -> list[NormalizableMeasurement]:
        """Returns the list of measurement included in the sum of square difference cost function"""
        return self.__measurements_in_sum_of_squares

    def measure(self) -> Array | float:
        """Sum of plain weighted measurements.

        Returns
        -------
        Array
            Returns the plain weighted sum.

        """
        values = [m.measure() for m in self.__measurements]
        sum_meas: Array | float = 0.0
        for ii, w in enumerate(self.__weights):
            sum_meas += w * values[ii]
        if self.__sum_of_squares_options is not None:
            sum_square_diff: Array | float = 0.0
            values_in_sum_of_squares = [
                val for val, flag in zip(values, self.__sum_of_squares_options["meas_bool"]) if flag
            ]
            for meas_a, meas_b in itertools.combinations(values_in_sum_of_squares, 2):
                sum_square_diff += (meas_a - meas_b) ** 2
            sum_meas += self.__sum_of_squares_options["weight"] * sum_square_diff
        return sum_meas

    def calculate_normalized_scalar(self) -> float:
        """Sum of weighted measurements from normalized measurements.

        Returns
        -------
        Array
            Returns the normalized weighted sum.

        """
        values = [m.calculate_normalized_scalar() for m in self.__measurements]
        sum_meas = 0.0
        for ii, w in enumerate(self.__weights):
            sum_meas += w * values[ii]
        if self.__sum_of_squares_options is not None:
            values_in_sum_of_squares = [
                val for val, flag in zip(values, self.__sum_of_squares_options["meas_bool"]) if flag
            ]
            sum_square_diff = 0.0
            for meas_a, meas_b in itertools.combinations(values_in_sum_of_squares, 2):
                sum_square_diff += (meas_a - meas_b) ** 2
            sum_meas += self.__sum_of_squares_options["weight"] * sum_square_diff
        return float(sum_meas)

    def calculate_value_and_gradient(self) -> tuple[Array, Array] | tuple[float, Array]:
        """Sum of weighted measurements from gradient-ized measurements.

        Returns
        -------
        Array
            Returns the weighted sum wrt to gradients.
        Array
            Returns the sum of gradients.

        """
        # TODO: this statement has a condition now: it checks whether the measurement is a Differentiable:
        # is it still correct?
        values_and_gradients = [
            m.calculate_value_and_gradient() for m in self.__measurements if isinstance(m, Differentiable)
        ]
        sum_meas = jnp.array(0)
        sum_grads = jnp.zeros_like(values_and_gradients[0][1])
        for ii, w in enumerate(self.__weights):
            sum_meas += w * values_and_gradients[ii][0]
            sum_grads += w * values_and_gradients[ii][1]
        if self.__sum_of_squares_options is not None:
            values_and_gradients_in_sum_of_squares = [
                val_grad
                for val_grad, flag in zip(values_and_gradients, self.__sum_of_squares_options["meas_bool"])
                if flag
            ]
            sum_square_diff = jnp.array(0)
            grads_diff = jnp.zeros_like(values_and_gradients_in_sum_of_squares[0][1])

            for meas_a, meas_b in itertools.combinations(values_and_gradients_in_sum_of_squares, 2):
                sum_square_diff += (meas_a[0] - meas_b[0]) ** 2
                grads_diff += 2 * (meas_a[0] - meas_b[0]) * (meas_a[1] - meas_b[1])
            sum_meas += self.__sum_of_squares_options["weight"] * sum_square_diff
            sum_grads += self.__sum_of_squares_options["weight"] * grads_diff
        return float(sum_meas), sum_grads
