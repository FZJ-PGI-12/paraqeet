"""Class definition of the pulse smoothness. It follows the definition in
[Heeres2017], in particular Eqs. 21 of the supplementary material.

References
[Heeres2017] R. Heeres et al., Nat. Comm. 8, 94 (2017)
"""

from typing import override

import jax
import jax.numpy as jnp

from paraqeet.differentiable import Differentiable
from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.quantity import Array, Float
from paraqeet.signal.pwc_generator import PWCGenerator

jax.config.update("jax_enable_x64", True)


class Smoothness(NormalizableMeasurement, Differentiable):
    """Smoothness of a pulse. It follows the definition in
    Heeres et al., https://arxiv.org/abs/1608.02430 (2017), in particular
    Eqs. 23 and 24 of the supplementary material.
    """

    _pwc_generator: PWCGenerator

    def __init__(self, pwc_generator: PWCGenerator):
        """...

        Args:
            pwc_generator: The generator from which we extract the pulse.
        """
        self._pwc_generator = pwc_generator

    @override
    def get_value(self, times: Array) -> Float:
        """Return the normalized sum of consecutive square differences of the pulse.

        As the maximums difference is twice the maximum amplitude, the normalization
        factor is the number of piecewise constants minus 1 times the maximum
        difference squared.

        Args:
            times: Array of times.

        Returns:
            The normalized sum of consecutive square differences in the pulse.
        """
        pulse = self._pwc_generator.get_value(times)
        num_pwc = jnp.shape(pulse)[0]

        norm_coeff = (num_pwc - 1) * (2 * self._pwc_generator.max_amplitude) ** 2

        def get_squared_difference(index):
            """Squared difference between two consecutive bins in PWC pulse.

            Args:
                index: Index of the bin.
            """
            return jnp.abs(pulse[index] - pulse[index + 1]) ** 2

        indices = jnp.arange(0, num_pwc - 1)
        vmap_get_squared_difference = jax.vmap(get_squared_difference)
        return 1.0 - jnp.sum(vmap_get_squared_difference(indices)) / norm_coeff

    @override
    def measure(self, times: Array) -> Float:
        return self.get_value(times)

    @override
    def calculate_normalized_scalar(self, times: Array) -> Float:
        return self.get_value(times)

    @override
    def get_gradient(self, times: Array) -> Array:
        """Compute the gradient.

        Compute with respect to all parameters in the optimization map.
        For parameters that are not in the passed PWCGenerator the partial derivative
        is simply zero.

        Args:
            times: Array of times. Not accessed, but we leave it for consistency
                with the abstract get_gradient method.

        Returns:
            Gradient of shape (n_parameters,).
        """
        opt_pwc_params = self._pwc_generator.optimizable_parameters
        opt_params = self._pwc_generator.all_optimizable_parameters

        def get_partial_derivative(n, vec):
            """Derivatives of the smoothness measure for 3 cases: starting point, center and end point.

            Args:
                n: Location in the piecewise constant vector.
                vec: Piecewise constant vector.
            """
            num_pwc = vec.shape[0]
            res = (
                ((2 * vec[n]) - vec[n + 1] - vec[n - 1])
                * (1.0 - jnp.where(n == 0, 1.0, 0.0))
                * (1.0 - jnp.where(n == num_pwc - 1, 1.0, 0.0))
            )
            res += (vec[0] - vec[1]) * jnp.where(n == 0, 1.0, 0.0)
            res += (vec[num_pwc - 1] - vec[num_pwc - 2]) * jnp.where(n == num_pwc - 1, 1.0, 0.0)
            return -res

        if len(opt_params) > 0:
            grad_list = []
            for param in opt_params:
                num_pwc = param.get_value().shape[0]
                norm_coeff = (num_pwc - 1) * (2 * self._pwc_generator.max_amplitude) ** 2
                if id(param) in [id(pwc_param) for pwc_param in opt_pwc_params]:
                    n_vec = jnp.arange(num_pwc)
                    vmap_get_partial_derivative = jax.vmap(get_partial_derivative, in_axes=(0, None))
                    grad_list.append(2 * vmap_get_partial_derivative(n_vec, param.get_value()) / norm_coeff)
                else:
                    grad_list.append(jnp.zeros(num_pwc))
            gradient = jnp.concatenate(grad_list)
        else:
            gradient = jnp.empty((1, 0))  # this is purely conventional

        return gradient
