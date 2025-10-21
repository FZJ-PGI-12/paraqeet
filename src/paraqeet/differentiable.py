# TODO: implement abstract class providing a method get_gradient

# All signal generators are differentiable?
# Is a default implementation possible? Not yet
#  Subclasses of Measurement lacking the impelmentation of the gradient calculation are
# NOT Differentiables? Yes

from abc import ABC, abstractmethod

from paraqeet.quantity import Array


class Differentiable(ABC):
    """An abstract class for differentiable models.

    Subclasses must implement the calculate_gradient() method which would
    return the gradient of the model.
    """

    @abstractmethod
    def calculate_value_and_gradient(self) -> tuple[Array, Array] | tuple[float, Array]:
        """Calculate the gradient of the model.

        Returns
        -------
        list[float]
            List of gradients with respect to each optimizable parameter.

        """
        pass
