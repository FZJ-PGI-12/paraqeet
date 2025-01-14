"""Class definition for the Optimisable model."""

from abc import abstractmethod

from cthree.Quantity import Quantity


class Optimisable:
    """Optimisable parameter model.

    This interface must be implemented by any class that provides optimisable
    parameters. The optimiser will collect all parameters (by reference) and
    update their values.

    """

    _name: str = None
    _optimisableParameters: list[Quantity] = []

    @abstractmethod
    def getParameters(self) -> list[Quantity]:
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

    def setOptimisableParameters(self, params: list[Quantity]) -> None:
        """Set which parameters shall be considered during optimisation.

        All quantities that are not in the response of getParameters will
        be filtered out. This function is called by the optimiser before
        gradient based optimisation to tell the layers which gradients to
        compute.

        Parameters
        ----------
        params : List[cthree.Quantity]
            List of optimisable parameters to be set.

        """
        allParams = self.getParameters()
        self._optimisableParameters = [
            p for p in params if any([p is q for q in allParams])
        ]

    def _isOptimised(self, param: Quantity) -> bool:
        """Check if a parameter is being optimised.

        Should therefore be included in gradients.

        Parameters
        ----------
        param : cthree.Quantity
            Input parameter to be checked for whether it is optimised.

        Returns
        -------
        bool
            True if parameter is optimised.

        """
        return param in self._optimisableParameters
