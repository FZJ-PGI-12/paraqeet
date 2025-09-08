"""Class definition of the pulse smoothness. It follows the definition in
Heeres et al., https://arxiv.org/abs/1608.02430 (2017), in particular
Eqs. 23 and 24 of the supplementary material.
"""

from paraqeet.measurement.measurement import Measurement
from paraqeet.signal.pwc_generator import PWCGenerator
from paraqeet.quantity import Array, Quantity
from paraqeet.optimisation_map import OptimisationMap

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class Smoothness(Measurement):
    """Smoothness of a pulse. It follows the definition in
    Heeres et al., https://arxiv.org/abs/1608.02430 (2017), in particular
    Eqs. 23 and 24 of the supplementary material.

    Parameters
    ----------
    pwc_generator: PWCGenerator
        The generator from which we extract the pulse.
    optmap: OptimisationMap
        The object that stores all optimisable parameters. It is needed
        to compute the gradient correctly.
    times: Array
        One-dimensional vector of timestamps.
    """

    _pwc_generator: PWCGenerator
    __optmap: OptimisationMap

    def __init__(self, pwc_generator: PWCGenerator, optmap: OptimisationMap):
        super().__init__(pwc_generator.tlist)
        self.__optmap = optmap
        self._pwc_generator = pwc_generator

    def get_parameters(self) -> list[Quantity]:
        """Returns an empty list."""
        return []

    def measure_normalised_scalar(self) -> float:
        """Returns the normalized sum of consecutive square differences of the pulse.
        As the maximums difference is twice the maximum amplitude, the normalization
        factor is the number of piecewise constants minus 1 time sthe maximum
        difference squared.

        Returns
        -------
        float
            The normalized sum of consecutive square differences in the pulse.
        """
        pulse = self._pwc_generator.generate_signal(self._times)
        num_pwc = jnp.shape(pulse)[0]

        norm_coeff = (num_pwc - 1) * (2 * self._pwc_generator.max_amplitude) ** 2

        def get_squared_difference(index: int):
            return jnp.abs(pulse[index] - pulse[index + 1]) ** 2

        indices = jnp.arange(0, num_pwc - 1)
        vmap_get_squared_difference = jax.vmap(get_squared_difference)
        return float(1.0 - jnp.sum(vmap_get_squared_difference(indices)) / norm_coeff)

    def measure_with_gradient(self) -> tuple[float, Array]:
        """Measure with gradient.

        Compute the measurement value as in measure_normalised_scalar()
        and the gradient with respect to all parameters in the optimisation map.
        For parameters that are not in the passed PWCGenerator the partial derivative
        if simply zero.

        Returns
        -------
        Tuple[float, Array]
            Tuple of function value as bare float and gradient of shape (n_parameters,)

        """
        opt_pwc_params = self._pwc_generator.optimisable_parameters
        opt_params = self.__optmap.get_all_parameters()
        # This cannot work. You should create
        # a function that returns the optimisable parameters, but only of PWC since the
        # gradient is only nonzero with respect to these. So there must be a way to flag
        # if these are parameters of the pwc passed to the class.

        def get_partial_derivative(n: int, vec: Array):
            num_pwc = vec.shape[0]
            res = (
                ((2 * vec[n]) - vec[n + 1] - vec[n - 1])
                * (1.0 - jnp.where(n == 0, 1.0, 0.0))
                * (1.0 - jnp.where(n == num_pwc - 1, 1.0, 0.0))
            )
            res += (vec[1] - vec[0]) * jnp.where(n == 0, 1.0, 0.0)
            res += (vec[num_pwc - 1] - vec[num_pwc - 2]) * jnp.where(n == num_pwc - 1, 1.0, 0.0)
            return res

        if len(opt_params) > 0:
            grad_list = []
            for param in opt_params:
                num_elements = param.get_value().shape[0]
                if id(param) in [id(pwc_param) for pwc_param in opt_pwc_params]:
                    n_vec = jnp.arange(num_elements)
                    vmap_get_partial_derivative = jax.vmap(get_partial_derivative, in_axes=(0, None))
                    grad_list.append(vmap_get_partial_derivative(n_vec, param.get_value()))
                else:
                    grad_list.append(jnp.zeros(num_elements))
            gradient = jnp.concatenate(grad_list)
        else:
            gradient = jnp.empty((1, 0))  # this is purely conventional

        return self.measure_normalised_scalar(), gradient
