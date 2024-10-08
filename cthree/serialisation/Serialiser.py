class Serialiser:
    """
    Interface for any class that can read and write configurations to a persistent format, e.g. a file. This can be used
    for the state of an optimisation or the setup of the layers.
    """
    def save(self, data: dict) -> None:
        raise NotImplementedError()

    def load(self) -> dict:
        raise NotImplementedError()
