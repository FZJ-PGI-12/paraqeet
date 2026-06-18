"""Class definition of the Weighted Sum Goal model."""

import jax.numpy as jnp
import numpy as np

from paraqeet.differentiable import Differentiable
from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.measurement.state_transfer_fidelity import (
    StateTransferFidelityGRAPE,
)
from paraqeet.propagation.differentiable_propagation import DifferentiablePropagation
from paraqeet.quantity import Array
from paraqeet.signal.pwc_generator import PWCGenerator


class GOATOverGRAPE(NormalizableMeasurement, Differentiable):
    """Combine GRAPE propagation with analytic gradients of GOAT via chain rule.

    Parameters
    ----------
    measurement : StateTransferFidelityGRAPE
        A StateTransferFidelityGRAPE measurement.
    propagation: DifferentiablePropagation
        Propagation method used for the optimization. Used to determine the time grid.
    generators: PWCGenerator | list[PWCGenerator]
        A PWCGenerator or a list of PWCGenerators that are used for propagation.
    """

    _measurement: StateTransferFidelityGRAPE
    _gens: list[PWCGenerator]
    _propagation: DifferentiablePropagation

    def __init__(
        self,
        measurement: StateTransferFidelityGRAPE,
        propagation: DifferentiablePropagation,
        generators: PWCGenerator | list[PWCGenerator],
    ):
        self._measurement = measurement
        self._propagation = propagation
        self._gens = generators if isinstance(generators, list) else [generators]
        for gen in self._gens:
            gen.set_optimizable_parameters(gen.get_parameters())

    def _pad_with_zeros(self, grad: Array, gen_num: int) -> Array:
        """Pad gradient with zeros depending on the subsystem number and number of PWC pixels in the pulses.

        Parameters
        ----------
        grad : Array
            Gradient from a subsystem.
        gen_num : int
            Subsystem number, also determined by the generator order.

        Returns
        -------
        Array
            Return padded gradient vector.
        """
        num_pixels, num_params = grad.shape
        padded_grad = np.zeros((0, num_params))
        for i in range(len(self._gens)):
            if i == gen_num:
                padded_grad = np.append(padded_grad, grad, axis=0)
            else:
                padded_grad = np.append(padded_grad, np.zeros((num_pixels, num_params)), axis=0)
        return jnp.array(padded_grad)

    def _construct_interpolated_times(self, times: Array):
        # Construct the same time grid as propagation to evaluate control gradients
        interp_times = jnp.array([])
        for ti in range(1, len(times)):
            t_interpolated, dt = self._propagation._construct_times(times, ti)
            interp_times = jnp.append(interp_times, t_interpolated, axis=0)

        interp_times = jnp.append(interp_times, interp_times[-1] + dt)
        return interp_times, dt

    def measure(self, times: Array) -> Array | float:
        """Sum of plain weighted measurements.

        Returns
        -------
        Array
            Returns the plain weighted sum.

        """
        grape = self._measurement
        for gen in self._gens:
            gen._update_inphase_and_outofphase()

        interp_times, _ = self._construct_interpolated_times(times)

        return grape.measure(times=interp_times)

    def calculate_normalized_scalar(self, times: Array | float) -> float:
        """Passthrough the measurement.

        Returns
        -------
        Array
            Returns the normalized weighted sum.

        """
        grape = self._measurement
        for gen in self._gens:
            gen._update_inphase_and_outofphase()
        return grape.calculate_normalized_scalar(times=times)

    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array] | tuple[float, Array]:
        """Compute gradients with GRAPE and use the chain rule
        to provide the gradients for the optimizer.

        Returns
        -------
        Array
            Function value.
        Array
            Gradients.

        """
        # TODO: Fix typing
        grape = self._measurement
        for gen in self._gens:
            gen._update_inphase_and_outofphase()

        interp_times, dt = self._construct_interpolated_times(times)
        time_grid = interp_times[:-1] + dt / 2

        control_gradients: list[Array] = []
        for gen_num, gen in enumerate(self._gens):
            control_gradients.append(self._pad_with_zeros(gen._get_partial_derivatives(time_grid), gen_num))
        control_gradients_arr = jnp.hstack(control_gradients)

        # Evaluate GRAPE gradients
        function_value, grape_gradients = grape.get_value_and_gradient(interp_times)

        # Reconstruct GOAT gradients
        goat_gradients = control_gradients_arr.T @ grape_gradients
        return function_value, goat_gradients
