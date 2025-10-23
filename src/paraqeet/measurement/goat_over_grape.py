"""Class definition of the Weighted Sum Goal model."""

from paraqeet.differentiable import Differentiable
from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.measurement.state_transfer_fidelity import (
    StateTransferFidelityGRAPE,
)
from paraqeet.quantity import Array
from paraqeet.signal.pwc_generator import PWCGenerator


class GOATOverGRAPE(NormalizableMeasurement, Differentiable):
    """Combine GRAPE propagation with analytic gradients of GOAT via chain rule.

    Parameters
    ----------
    measurement : List[Measurement]
        List of measurements.
    weights : Array
        List of weights.

    Raises
    ------
    ConfigurationException
        If number of measurements and weights are incompatible.
    UserWarning
        If the given weights are not normalized.

    """

    __measurement: StateTransferFidelityGRAPE
    __gen: PWCGenerator

    def __init__(self, measurement: StateTransferFidelityGRAPE, gen: PWCGenerator):
        self.__measurement = measurement
        self.__gen = gen
        gen.set_optimizable_parameters(gen.get_parameters())

    def measure(self, times: Array) -> Array | float:
        """Sum of plain weighted measurements.

        Returns
        -------
        Array
            Returns the plain weighted sum.

        """
        grape = self.__measurement
        self.__gen._update_inphase_and_outofphase()
        return grape.measure(times=times)

    def calculate_normalized_scalar(self, times: Array) -> float:
        """Passthrough the measurement.

        Returns
        -------
        Array
            Returns the normalized weighted sum.

        """
        grape = self.__measurement
        self.__gen._update_inphase_and_outofphase()
        return grape.calculate_normalized_scalar(times=times)

    def value_and_gradient(self, times: Array) -> tuple[Array, Array] | tuple[float, Array]:
        """Compute gradients with GRAPE and use the chain rule
        to provide the gradients for the optimizer.

        Returns
        -------
        Array
            Function value.
        Array
            Gradients.

        """
        grape = self.__measurement
        self.__gen._update_inphase_and_outofphase()
        control_gradients = self.__gen._get_partial_derivatives()
        function_value, grape_gradients = grape.value_and_gradient(times)
        goat_gradients = control_gradients.T @ grape_gradients
        return function_value, goat_gradients
