"""Class definition of the Weighted Sum Goal model."""

from paraqeet.measurement.measurement import Measurement
from paraqeet.measurement.state_transfer_fidelity import StateTransferFidelityGRAPE
from paraqeet.quantity import Array
from paraqeet.signal.pwc_generator import PWCGenerator


class GOATOverGRAPE(Measurement):
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
        gen.set_optimisable_parameters(gen.get_parameters())

    def measure(self) -> Array | float:
        """Sum of plain weighted measurements.

        Returns
        -------
        Array
            Returns the plain weighted sum.

        """
        grape = self.__measurement
        self.__gen._update_inphase_and_quadrature()
        return grape.measure()

    def measure_normalised_scalar(self) -> float:
        """Passthrough the measurement.

        Returns
        -------
        Array
            Returns the normalized weighted sum.

        """
        grape = self.__measurement
        self.__gen._update_inphase_and_quadrature()
        return grape.measure_normalised_scalar()

    def measure_with_gradient(self) -> tuple[float, Array]:
        """Compute gradients with GRAPE and use the chain rule
        to provide the gradients for the optimiser.

        Returns
        -------
        Array
            Function value.
        Array
            Gradients.

        """
        grape = self.__measurement
        self.__gen._update_inphase_and_quadrature()
        control_gradients = self.__gen._get_partial_derivatives()
        function_value, grape_gradients = grape.measure_with_gradient()
        goat_gradients = control_gradients.T @ grape_gradients
        return function_value, goat_gradients
