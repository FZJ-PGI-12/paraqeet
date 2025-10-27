"""Class definition of the Weighted Sum Goal model."""

import jax.numpy as jnp
import numpy as np

from paraqeet.exceptions import ConfigurationException
from paraqeet.measurement.measurement import Measurement
from paraqeet.measurement.state_transfer_fidelity import StateTransferFidelityGRAPE
from paraqeet.quantity import Array
from paraqeet.signal.pwc_generator import PWCGenerator


class GOATOverGRAPE(Measurement):
    """Combine GRAPE propagation with analytic gradients of GOAT via chain rule.

    *Note - currently only works with one PWCGenerator per subsystem.*

    Parameters
    ----------
    measurement : StateTransferFidelityGRAPE
        A StateTransferFidelityGRAPE measurement.
    generators: PWCGenerator | list[PWCGenerator]
        A PWCGenerator or a list of PWCGenerators that are used for propagation.
    generators_order: list[int]
        Specify the subsystem number (starting with 0) the corresponding generator is associated with.
        This is required to correctly order the gradients obtained from GRAPE.
    """

    __measurement: StateTransferFidelityGRAPE
    _gens: list[PWCGenerator]
    _gens_order: list[int]
    _num_pwc_pixels: list[int]

    def __init__(
        self,
        measurement: StateTransferFidelityGRAPE,
        generators: PWCGenerator | list[PWCGenerator],
        generators_order: list[int],
    ):
        self.__measurement = measurement
        self._gens = generators if isinstance(generators, list) else [generators]
        for gen in self._gens:
            gen.set_optimisable_parameters(gen.get_parameters())
        self._gens_order = generators_order

        if len(self._gens) != len(self._gens_order):
            raise ConfigurationException(
                f"No. of generators and generators_order should be the same. \
                Got len(generators)={len(self._gens)} and len(generators_order) = {len(self._gens_order)}."
            )

        # we generate the num_pwc_pixel in the ascending order of subsystem number (0, 1, 2 ...)
        self._num_pwc_pixels = [self._gens[i].get_number_of_pwc_pixels() for i in self._gens_order]

    def get_parameters(self):
        """Return measurement specific parameters. This class does not contain any optimisable parameters."""
        return []

    def pad_with_zeros(self, grad: Array, subsys_num: int) -> Array:
        """Pad gradient with zeros depending on the subsystem number and number of PWC pixels in the pulses.

        Parameters
        ----------
        grad : Array
            _description_
        subsys_num : int
            _description_

        Returns
        -------
        Array
            _description_
        """
        num_params = grad.shape[1]
        padded_grad = np.zeros((0, num_params))
        for i, n_pixel in enumerate(self._num_pwc_pixels):
            if i == subsys_num:
                padded_grad = np.append(padded_grad, grad, axis=0)
            else:
                padded_grad = np.append(padded_grad, np.zeros((2 * n_pixel, num_params)), axis=0)
        return jnp.array(padded_grad)

    def measure(self) -> Array | float:
        """Sum of plain weighted measurements.

        Returns
        -------
        Array
            Returns the plain weighted sum.

        """
        grape = self.__measurement
        for gen in self._gens:
            gen._update_inphase_and_outofphase()
        return grape.measure()

    def measure_normalised_scalar(self) -> float:
        """Passthrough the measurement.

        Returns
        -------
        Array
            Returns the normalized weighted sum.

        """
        grape = self.__measurement
        for gen in self._gens:
            gen._update_inphase_and_outofphase()
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
        for gen in self._gens:
            gen._update_inphase_and_outofphase()

        control_gradients: list[Array] = []
        for subsys_num, gen in zip(self._gens_order, self._gens):
            control_gradients.append(self.pad_with_zeros(gen._get_partial_derivatives(), subsys_num))
        control_gradients_arr = jnp.hstack(control_gradients)

        function_value, grape_gradients = grape.measure_with_gradient()
        goat_gradients = control_gradients_arr.T @ grape_gradients
        return function_value, goat_gradients
