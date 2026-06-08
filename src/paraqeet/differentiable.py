# TODO: implement abstract class providing a method get_gradient

# All signal generators are differentiable?
# Is a default implementation possible? Not yet
#  Subclasses of Measurement lacking the impelmentation of the gradient calculation are
# NOT Differentiables? Yes

from abc import ABC, abstractmethod

from paraqeet.quantity import Array, Float


class Differentiable(ABC):
    """An abstract class for differentiable models.

    Subclasses must implement the get_value, get_gradient and get_value_and_gradient
    methods. Note that in some cases the gradient gets computed together with the
    value and thus it is more natural to instantiate a get_gradient method
    using the result of get_value_and_gradient. This is the reason why all
    three methods are taken as abstract classes, so that one always thinks
    about the right implementation.
    """

    @abstractmethod
    def get_value(self, times: Array) -> Float | Array:
        """Calculate the value of the object at different times.

        Parameters
        ----------
            times: Array of times.

        Returns:
        ----------
            The value of the object.
        """
        pass

    @abstractmethod
    def get_gradient(self, times: Array) -> Array:
        """Calculate the gradient of the object at different times.

        Parameters
        ----------
            times: Array of times.

        Returns:
        ----------
            The gradient of the object.
        """
        pass

    @abstractmethod
    def get_value_and_gradient(self, times: Array) -> tuple:
        """Calculate the value and the gradient of the object at different times.

        Returns:
        ----------
            The value and the gradient of the object.
        """
        pass
