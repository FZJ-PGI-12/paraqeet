"""Custom Exception implementations."""


class IncompatibleLayersException(Exception):
    """Raise when incompatible layers transmission.

    Raised when a layer can not handle the form of the result of the
    previous layer. For example, a unitary fidelity will throw this if the
    propagation layer only provides the propagated state.

    """

    pass


class ConfigurationException(Exception):
    """Raise when configuration issue.

    Raised when a layer implementation was not properly configured before
    running it.

    """

    pass


class IncompatibleQuantityException(Exception):
    """Raise when incompatible quantity shape.

    Raised when a quantity has an unexpected shape, e.g. a vector quantity
    if a scalar was expected.

    """

    pass


class IncompatibleOptimisationMap(Exception):
    """Raise when incorrect number of quantities are specified.

    Raised when the number of quantities specified in optimisation map
    doesnt match the number of gradients computed.

    """

    pass


class SerialisationException(Exception):
    """Raised when reading or writing of a Quantity or an OptimisationMap fails."""

    pass
