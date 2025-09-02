"""Class definition of the pulse smoothness. It follows the definition in
Heeres et al., https://arxiv.org/abs/1608.02430 (2017), in particular
Eqs. 23 and 24 of the supplementary material.
"""

from paraqeet.measurement.measurement import Measurement
from paraqeet.signal.pwc_generator import PWCGenerator

import jax

jax.config.update("jax_enable_x64", True)


class Smoothness(Measurement):
    """Smoothness of a pulse. It follows the definition in
    Heeres et al., https://arxiv.org/abs/1608.02430 (2017), in particular
    Eqs. 23 and 24 of the supplementary material.

    Parameters
    ----------
    pwc_generator: PWCGenerator
        The generator from which we extract the pulse
    times: Array
        One-dimensional vector of timestamps.
    """

    _pwc_generator: PWCGenerator

    def __init__(self, pwc_generator: PWCGenerator):
        super().__init__(pwc_generator.tlist)
        self._pwc_generator = pwc_generator

    # def measure(self):
    #     pulse = self._pwc_generator.generate_signal(self._times)

    #     def get_squared_difference(index: int):
    #         return jnp.abs(pulse[index] - pulse[index + 1]) ** 2

    #     indices = jnp.arange(0, jnp.shape(pulse)[0])
    #     vmap_get_squared_difference = jax.vmap(get_squared_difference)
    #     # norm_coeff = 2 * self._pwc_generator.
    #     return jnp.sum(vmap_get_squared_difference(indices))
