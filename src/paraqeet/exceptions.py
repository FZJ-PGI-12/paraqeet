"""Custom Exception implementations."""


class IncompatibleLayersException(Exception):
    """Raised when incompatible layers are connected.

    Raised when a layer can not handle the form of the result of the
    previous layer. For example, a unitary fidelity will throw this if the
    propagation layer only provides the propagated state.
    """

    pass


class ConfigurationException(Exception):
    """Raised when there is a configuration issue.

    Raised when a layer implementation was not properly configured before
    running it.
    """

    pass


class IncompatibleQuantityException(Exception):
    """Raised when a quantity has an incompatible shape.

    Raised when a quantity has an unexpected shape, e.g. a vector quantity
    if a scalar was expected.
    """

    pass


class IncompatibleOptimizationMap(Exception):
    """Raised when an incorrect number of quantities is specified.

    Raised when the number of quantities specified in optimization map
    doesn't match the number of gradients computed.
    """

    pass


class SerializationException(Exception):
    """Raised when reading or writing of a Quantity or an OptimizationMap fails."""

    pass
