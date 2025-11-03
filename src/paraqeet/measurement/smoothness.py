"""Class definition of the pulse smoothness. It follows the definition in
[Heeres2017], in particular Eqs. 21 of the supplementary material.

References
[Heeres2017] R. Heeres et al., Nat. Comm. 8, 94 (2017)
"""

import jax
import jax.numpy as jnp

from paraqeet.differentiable import Differentiable
from paraqeet.measurement.measurement import NormalizableMeasurement
from paraqeet.quantity import Array
from paraqeet.signal.pwc_generator import PWCGenerator

jax.config.update("jax_enable_x64", True)


class Smoothness(NormalizableMeasurement, Differentiable):
    """Smoothness of a pulse. It follows the definition in
    Heeres et al., https://arxiv.org/abs/1608.02430 (2017), in particular
    Eqs. 23 and 24 of the supplementary material.

    Parameters
    ----------
    pwc_generator: PWCGenerator
        The generator from which we extract the pulse.
    times: Array
        One-dimensional vector of timestamps.
    """

    _pwc_generator: PWCGenerator

    def __init__(self, pwc_generator: PWCGenerator):
        # super().__init__(pwc_generator.tlist)
        self._pwc_generator = pwc_generator

    def measure(self, times: Array) -> Array | float:
        """Return measurement in the range [0, 1]."""
        return self.calculate_normalized_scalar(times)

    # TODO: This should depend on the internal time grid and not on the input time value.
    # This means that the `times` should just be a float.
    def calculate_normalized_scalar(self, times: Array | float) -> float:
        """Returns the normalized sum of consecutive square differences of the pulse.
        As the maximums difference is twice the maximum amplitude, the normalization
        factor is the number of piecewise constants minus 1 time sthe maximum
        difference squared.

        Returns
        -------
        float
            The normalized sum of consecutive square differences in the pulse.
        """
        pulse = self._pwc_generator.generate_signal(times)
        num_pwc = jnp.shape(pulse)[0]

        norm_coeff = (num_pwc - 1) * (2 * self._pwc_generator.max_amplitude) ** 2

        def get_squared_difference(index):
            """Squared difference between two consecutive bins in PWC pulse.

            Parameters
            ----------
            index : int
                index of the bin
            """
            return jnp.abs(pulse[index] - pulse[index + 1]) ** 2

        indices = jnp.arange(0, num_pwc - 1)
        vmap_get_squared_difference = jax.vmap(get_squared_difference)
        return float(1.0 - jnp.sum(vmap_get_squared_difference(indices)) / norm_coeff)

    def value_and_gradient(self, times: Array) -> tuple[Array, Array] | tuple[float, Array]:
        """Measure with gradient.

        Compute the measurement value as in measure_normalized_scalar()
        and the gradient with respect to all parameters in the optimization map.
        For parameters that are not in the passed PWCGenerator the partial derivative
        if simply zero.

        Returns
        -------
        Tuple[float, Array]
            Tuple of function value as bare float and gradient of shape (n_parameters,)

        """
        opt_pwc_params = self._pwc_generator.optimizable_parameters
        opt_params = self._pwc_generator.all_optimizable_parameters

        def get_partial_derivative(n, vec):
            """Derivatives of the smoothness measure for 3 cases: starting point, center and end point.

            Parameters
            ----------
            n : int
                location in the piecewise constant vector
            vec : Array
                piecewise constant vector
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

        return self.calculate_normalized_scalar(times), gradient
