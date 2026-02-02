"""Class definition of the Weighted Sum Goal model."""

import jax.numpy as jnp
import numpy as np

from paraqeet.exceptions import ConfigurationException
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

    *Note - currently only works with one PWCGenerator per subsystem.*

    Parameters
    ----------
    measurement : StateTransferFidelityGRAPE
        A StateTransferFidelityGRAPE measurement.
    propagation: DifferentiablePropagation
        Propagation method used for the optimization. Used to determine the time grid.
    generators: PWCGenerator | list[PWCGenerator]
        A PWCGenerator or a list of PWCGenerators that are used for propagation.
    generators_order: list[int]
        Specify the subsystem number (starting with 0) the corresponding generator is associated with.
        This is required to correctly order the gradients obtained from GRAPE.

    Raises
    ------
    ConfigurationException
        If number of generators and generator_order are not equal.
    """

    _measurement: StateTransferFidelityGRAPE
    _gens: list[PWCGenerator]
    _gens_order: list[int]
    _num_pwc_pixels: list[int]
    _propagation: DifferentiablePropagation

    def __init__(
        self,
        measurement: StateTransferFidelityGRAPE,
        propagation: DifferentiablePropagation,
        generators: PWCGenerator | list[PWCGenerator],
        generators_order: list[int],
    ):
        self._measurement = measurement
        self._propagation = propagation
        self._gens = generators if isinstance(generators, list) else [generators]
        for gen in self._gens:
            gen.set_optimizable_parameters(gen.get_parameters())
        self._gens_order = generators_order

        if len(self._gens) != len(self._gens_order):
            raise ConfigurationException(
                f"No. of generators and generators_order should be the same. \
                Got len(generators)={len(self._gens)} and len(generators_order) = {len(self._gens_order)}."
            )

        # we generate the num_pwc_pixel in the ascending order of subsystem number (0, 1, 2 ...)
        self._num_pwc_pixels = [self._gens[i].get_number_of_pwc_pixels() for i in self._gens_order]

    def _pad_with_zeros(self, grad: Array, subsys_num: int) -> Array:
        """Pad gradient with zeros depending on the subsystem number and number of PWC pixels in the pulses.

        Parameters
        ----------
        grad : Array
            Gradient from a subsystem.
        subsys_num : int
            Subsystem number, also determined by the generator order.

        Returns
        -------
        Array
            Return padded gradient vector.
        """
        num_params = grad.shape[1]
        padded_grad = np.zeros((0, num_params))
        for i, n_pixel in enumerate(self._num_pwc_pixels):
            if i == subsys_num:
                padded_grad = np.append(padded_grad, grad, axis=0)
            else:
                padded_grad = np.append(padded_grad, np.zeros((2 * n_pixel, num_params)), axis=0)
        return jnp.array(padded_grad)

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
        return grape.measure(times=times)

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

    def _construct_times(self, time, ti):
        """Construct one-dimensional vector of time.

        In specified resolution at a snapshot.

        *NOTE - This function is taken from `ScipyExpm._construct_times`
        to match propagation and measurement time grids.*

        Parameters
        ----------
        time: Array
            Array of timesteps.
        ti: int
            Snapshot of the time at a current step

        Returns
        -------
        Array
            Array of timestamps in specified resolution.
        int
            Difference in time step.

        """
        t0 = time[ti - 1]
        t1 = time[ti]
        steps = int(jnp.ceil((t1 - t0) * self._propagation._res))
        times = jnp.linspace(t0, t1, steps, endpoint=False)
        if steps < 2:
            dt = t1 - t0
        else:
            dt = times[1] - times[0]
        return times, dt

    # TODO: adjust methods calling value_and_gradient
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

        # Construct the same time grid as propagation to evaluate control gradients
        interp_times = jnp.array([])
        for ti in range(1, len(times)):
            t_interpolated, dt = self._construct_times(times, ti)
            interp_times = jnp.append(interp_times, t_interpolated, axis=0)

        time_grid = interp_times[:-1] + dt / 2

        control_gradients: list[Array] = []
        for subsys_num, gen in zip(self._gens_order, self._gens):
            control_gradients.append(self._pad_with_zeros(gen._get_partial_derivatives(time_grid), subsys_num))
        control_gradients_arr = jnp.hstack(control_gradients)

        # Evaluate GRAPE gradients
        function_value, grape_gradients = grape.get_value_and_gradient(interp_times)

        # Reconstruct GOAT gradients
        goat_gradients = control_gradients_arr.T @ grape_gradients
        return function_value, goat_gradients
