import json

from paraqeet.exceptions import SerialisationException
from paraqeet.serialisation.serialiser import Serialiser


class JSONFileSerialiser(Serialiser):
    """Writes data into and read data from JSON files in a human-readable format."""

    __COMMENT_KEY = "__comment"
    __file: str

    def __init__(self, file: str):
        self.__file = file

    def save(self, data: dict, comment: str | None = None) -> None:
        """Saves the data and the optional comment to the JSON file that was specified in the constructor."""
        # The comment is simply stored in the same dict
        if comment:
            data[self.__COMMENT_KEY] = comment
        with open(self.__file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self) -> dict:
        """Loads and returns the data from JSON file"""
        with open(self.__file) as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise SerialisationException("File does not contain a dictionary.")

            if self.__COMMENT_KEY in data:
                del data[self.__COMMENT_KEY]
            return data

    def load_comment(self) -> str | None:
        """Loads and returns the comment from the JSON file."""
        with open(self.__file) as f:
            data = json.load(f)
            return data[self.__COMMENT_KEY] if self.__COMMENT_KEY in data else None
