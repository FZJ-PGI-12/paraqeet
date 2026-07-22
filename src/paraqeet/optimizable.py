"""Interface for classes that expose optimizable parameters to the optimizer."""

from abc import ABC, abstractmethod

from paraqeet.quantity import Quantity


class Optimizable(ABC):
    """Optimizable parameter model.

    This interface must be implemented by any class that provides optimizable
    parameters. The optimizer will collect all parameters (by reference) and
    update their values.
    """

    _name: str | None = None
    _optimizable_parameters: list[Quantity] = []
    _all_optimizable_parameters: list[Quantity] = []

    @abstractmethod
    def get_parameters(self) -> list[Quantity]:
        """Return all parameters of this class that can be optimized.

        Raises:
            NotImplementedError: Subclasses derived from this class must
                implement this method.
        """
        pass

    @property
    def name(self) -> str | None:
        """Get the name of the parameter.

        Returns:
            Name of the parameter.
        """
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:
        """Set the name of the parameter.

        Args:
            name: Value of the name to be set.
        """
        self._name = name

    @property
    def optimizable_parameters(self) -> list[Quantity]:
        """Get the optimizable parameters.

        Returns:
            The list of optimizable parameters associated with the object.
        """
        return self._optimizable_parameters

    @property
    def all_optimizable_parameters(self) -> list[Quantity]:
        """Get the optimizable parameters.

        Returns:
            The list of all the optimizable parameters considered in the
            optimization.
        """
        return self._all_optimizable_parameters

    def __repr__(self) -> str:
        """Magic method for human readable representation."""
        return self.__str__()

    def __str__(self) -> str:
        """Magic method for human readable string representation."""
        return self._name or str(self.__class__)

    def set_optimizable_parameters(self, params: list[Quantity]) -> None:
        """Set which parameters associated with the object shall be considered
        during optimization.

        All quantities that are not in the response of get_parameters will
        be filtered out. This function is called by the optimizer before
        gradient based optimization to tell the layers which gradients to
        compute.

        Args:
            params: List of optimizable parameters to be set.
        """
        own_params = self.get_parameters()
        self._optimizable_parameters = [p for p in params if any([p is q for q in own_params])]

    def set_all_optimizable_parameters(self, all_params: list[Quantity]) -> None:
        """Set all optimizable parameters in the optimization.

        Args:
            all_params: List of optimizable parameters to be set.
        """
        self._all_optimizable_parameters = all_params

    def _is_optimized(self, param: Quantity) -> bool:
        """Check if a parameter is being optimized.

        Should therefore be included in gradients.

        Args:
            param: Input parameter to be checked for whether it is optimized.

        Returns:
            True if parameter is optimized.
        """
        return id(param) in [id(opt_param) for opt_param in self._optimizable_parameters]
