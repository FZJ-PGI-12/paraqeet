# TODO: implement abstract class providing a method get_gradient

# All signal generators are differentiable?
# Is a default implementation possible? Not yet
#  Subclasses of Measurement lacking the impelmentation of the gradient calculation are
# NOT Differentiables? Yes

from abc import ABC, abstractmethod

from paraqeet.quantity import Array, Float


class Differentiable(ABC):
    """An abstract class for differentiable models.

    Subclasses must implement the value_and_gradient() method which would
    return the gradient of the model.
    """

    @abstractmethod
    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array] | tuple[Float, Array]:
        """Calculate the value and the gradient of the model.

        Returns
        -------
        tuple[Array, Array] | tuple[Float, Array]
            The value and the gradient of the model.

        """
        pass
