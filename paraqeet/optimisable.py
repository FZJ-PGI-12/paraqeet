"""Class definition for the Optimisable model."""

from abc import abstractmethod

from paraqeet.quantity import Quantity


class Optimisable:
    """Optimisable parameter model.

    This interface must be implemented by any class that provides optimisable
    parameters. The optimiser will collect all parameters (by reference) and
    update their values.

    """

    _name: str | None = None
    _optimisable_parameters: list[Quantity] = []

    @abstractmethod
    def get_parameters(self) -> list[Quantity]:
        """Return all parameters of this class that can be optimised.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    @property
    def name(self) -> str | None:
        """Get the name of the parameter.

        Returns
        -------
        str | None
            Name of the parameter.

        """
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:
        """Set the name of the parameter.

        Parameters
        ----------
        name : str
            Value of the name to be set.

        """
        self._name = name

    def __repr__(self):
        """Magic method for human readable representation."""
        return self.__str__()

    def __str__(self):
        """Magic method for human readable string representation."""
        return self._name or str(self.__class__)

    def set_optimisable_parameters(self, params: list[Quantity]) -> None:
        """Set which parameters shall be considered during optimisation.

        All quantities that are not in the response of getParameters will
        be filtered out. This function is called by the optimiser before
        gradient based optimisation to tell the layers which gradients to
        compute.

        Parameters
        ----------
        params : List[Quantity]
            List of optimisable parameters to be set.

        """
        allParams = self.get_parameters()
        self._optimisable_parameters = [p for p in params if any([p is q for q in allParams])]

    def _is_optimised(self, param: Quantity) -> bool:
        """Check if a parameter is being optimised.

        Should therefore be included in gradients.

        Parameters
        ----------
        param : paraqeet.quantity
            Input parameter to be checked for whether it is optimised.

        Returns
        -------
        bool
            True if parameter is optimised.

        """
        return id(param) in [id(opt_param) for opt_param in self._optimisable_parameters]
