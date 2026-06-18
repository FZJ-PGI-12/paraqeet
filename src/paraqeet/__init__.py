"""paraqeet: A quantum optimal control toolkit with simple parameter management."""

import jax

from paraqeet.optimization_map import OptimizationMap as OptimizationMap
from paraqeet.quantity import Array as Array
from paraqeet.quantity import Quantity as Quantity

jax.config.update("jax_enable_x64", True)
