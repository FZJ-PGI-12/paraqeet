# TODO: implement abstract class providing a method get_gradient

# All signal generators are differentiable?
# Is a default implementation possible? Not yet
#  Subclasses of Measurement lacking the impelmentation of the gradient calculation are
# NOT Differentiables? Yes

from abc import ABC, abstractmethod

from jax import vmap

from paraqeet.quantity import Array


class Differentiable(ABC):
    """An abstract class for differentiable models.

    Subclasses must implement the value_and_gradient() method which would
    return the gradient of the model.
    """

    def value_and_gradient(self, times: Array) -> tuple[Array, Array] | tuple[float, Array]:
        """Calculate the gradient of the model.

        Returns
        -------
        tuple[Array, Array] | tuple[float, Array]
            The value and the gradient of the model.

        """
        return vmap(self.value_and_gradient_at_timestep)(times)  # type: ignore

    @abstractmethod
    def value_and_gradient_at_timestep(self, time: float) -> tuple[Array, Array] | tuple[float, Array]:
        """Calculate the gradient of the model for one timestep.

        Returns
        -------
        tuple[Array, Array] | tuple[float, Array]
            The value and the gradient of the model.

        """
        pass
