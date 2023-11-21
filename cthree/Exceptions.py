class IncompatibleLayersException(Exception):
    """Raised when a layer can not handle the form of the result of the previous layer. For example, a unitary fidelity
    will throw this if the propagation layer only provides the propagated state."""

    pass


class ConfigurationException(Exception):
    """
    Raised when a layer implementation was not properly configured before running it.
    """

    pass
