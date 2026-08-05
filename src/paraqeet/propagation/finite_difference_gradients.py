"""Compute gradients of a propagation method by finite differences of its parameters."""

from typing import override

import jax.numpy as jnp
import numpy as np

from paraqeet.differentiable import Differentiable
from paraqeet.exceptions import ConfigurationException
from paraqeet.optimization_map import OptimizationMap
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array


class FiniteDifferenceGradients(Differentiable):
    r"""Wraps a Propagation object to add gradient computation by central finite differences.

    Every parameter of the optimization map is displaced by a small step in both directions, and
    the propagation is solved again for each of them,

        .. math::
            \frac{\partial \ket{\psi(t)}}{\partial \alpha} \approx
            \frac{\ket{\psi_{\alpha + h}(t)} - \ket{\psi_{\alpha - h}(t)}}{2h}.

    Note:
        This method is meant for verification only. It costs two full propagations per parameter,
        which is prohibitive for a pulse with many parameters. Use it to check the gradient of one of
        the analytic methods, and optimize with those.
    """

    _prop: Propagation
    _optimization_map: OptimizationMap
    _epsilon: float

    def __init__(
        self,
        propagation: Propagation,
        optimization_map: OptimizationMap,
        epsilon: float = 1e-6,
    ) -> None:
        """
        Args:
            propagation: Any propagation object.
            optimization_map: The optimization map holding the parameters to differentiate for.
                The gradient follows its order of parameters, like the analytic methods do.
            epsilon: Size of the displacement of a parameter, relative to the largest of its
                values, or to the range between its bounds if all of them are zero.

        Raises:
            ConfigurationException: If the displacement is not positive.
        """
        self._prop = propagation
        if epsilon <= 0.0:
            raise ConfigurationException("The displacement of the finite differences has to be positive.")
        self._optimization_map = optimization_map
        self._epsilon = epsilon

    @property
    def epsilon(self) -> float:
        """Return the displacement of a parameter, relative to its range."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon: float) -> None:
        """Set the displacement of a parameter, relative to its range.

        Raises:
            ConfigurationException: If the displacement is not positive.
        """
        if epsilon <= 0.0:
            raise ConfigurationException("The displacement of the finite differences has to be positive.")
        self._epsilon = epsilon

    @override
    def get_value(self, times: Array) -> Array:
        """Return the solution of the equations of motion from the propagation method.

        Args:
            times: Array of times.

        """
        return self._prop.get_value(times)

    @override
    def get_gradient(self, times: Array) -> Array:
        """Return the central difference of the propagated state/propagator for every parameter.

        The parameters are displaced one after the other, in the order of the optimization map.

        Args:
            times: Array of times.

        Returns:
            The gradient with shape ``(n_times, n_params) + state.shape``.

        Raises:
            ConfigurationException: If the optimization map holds no parameters.
        """
        parameters = self._optimization_map.get_all_parameters()
        if len(parameters) == 0:
            raise ConfigurationException("The optimization map of FiniteDifferenceGradients is empty.")

        gradients = []
        for parameter in parameters:
            values = np.array(parameter.get_value())
            magnitude = float(np.max(np.abs(values)))
            if magnitude == 0.0:
                magnitude = float(np.max(np.array(parameter.get_scale())))
            step = self._epsilon * magnitude

            try:
                parameter.set_value(values + step)
                plus = np.array(self.get_value(times))

                parameter.set_value(values - step)
                minus = np.array(self.get_value(times))
            finally:
                parameter.set_value(values)

            gradients.append((plus - minus) / (2 * step))

        return jnp.stack(gradients, axis=1)
