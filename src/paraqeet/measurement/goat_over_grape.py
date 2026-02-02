"""Class definition of the Weighted Sum Goal model."""

from paraqeet.differentiable import Differentiable
from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.measurement.state_transfer_fidelity import (
    StateTransferFidelityGRAPE,
)
from paraqeet.propagation.differentiable_propagation import DifferentiablePropagation
from paraqeet.quantity import Array
from paraqeet.signal.pwc_generator import PWCGenerator

import jax.numpy as jnp


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

    _measurement: StateTransferFidelityGRAPE
    _gen: PWCGenerator
    _propagation: DifferentiablePropagation

    def __init__(
        self, measurement: StateTransferFidelityGRAPE, gen: PWCGenerator, propagation: DifferentiablePropagation
    ):
        self._measurement = measurement
        self._gen = gen
        self._propagation = propagation
        gen.set_optimizable_parameters(gen.get_parameters())

    def measure(self, times: Array) -> Array | float:
        """Sum of plain weighted measurements.

        Returns
        -------
        Array
            Returns the plain weighted sum.

        """
        grape = self._measurement
        self._gen._update_inphase_and_outofphase()
        return grape.measure(times=times)

    def calculate_normalized_scalar(self, times: Array | float) -> float:
        """Passthrough the measurement.

        Returns
        -------
        Array
            Returns the normalized weighted sum.

        """
        grape = self._measurement
        self._gen._update_inphase_and_outofphase()
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
        self._gen._update_inphase_and_outofphase()

        # Construct the same time grid as propagation to evaluate control gradients
        interp_times = jnp.array([])
        for ti in range(1, len(times)):
            t_interpolated, dt = self._construct_times(times, ti)
            interp_times = jnp.append(interp_times, t_interpolated, axis=0)

        time_grid = interp_times[:-1] + dt / 2
        control_gradients = self._gen._get_partial_derivatives(time_grid)

        # Evaluate GRAPE gradients
        function_value, grape_gradients = grape.get_value_and_gradient(interp_times)

        # Reconstruct GOAT gradients
        goat_gradients = control_gradients.T @ grape_gradients
        return function_value, goat_gradients
