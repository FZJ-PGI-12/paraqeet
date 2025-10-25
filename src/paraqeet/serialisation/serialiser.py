class Serialiser:
    """
    Interface for any class that can read and write configurations to a persistent format, e.g. a file. This can be used
    for the state of an optimisation or the setup of the layers.
    """

    def save(self, data: dict, comment: str | None = None) -> None:
        """
        Saves data to a persistent format. The actual format depends on the implementation.

        Parameters
        ----------
        data: dict
            the data to be exported
        comment: str
            Optional comment to be stored with the data, for example a description of the data. Implementations have
            to decide how to store the comment.
        -------

        """
        raise NotImplementedError()

    def load(self) -> dict:
        """Loads and returns data that was previously saved."""
        raise NotImplementedError()

    def load_comment(self) -> str | None:
        """Loads and returns the comment, if any, that was previously saved with the data. Returns None if no comment
        was saved.

        Returns
            The comment, or None if no comment was saved.
        """
        raise NotImplementedError()
