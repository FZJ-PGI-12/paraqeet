"""Class definition of JSONFileSerializer."""

import json

from paraqeet.exceptions import SerializationException
from paraqeet.serialization.serializer import Serializer


class JSONFileSerializer(Serializer):
    """Write data into and read data from JSON files in a human-readable format."""

    _COMMENT_KEY = "_comment"
    _file: str

    def __init__(self, file: str) -> None:
        self._file = file

    def save(self, data: dict, comment: str | None = None) -> None:
        """Save the data and the optional comment to the JSON file that was specified in the constructor."""
        # The comment is simply stored in the same dict
        if comment:
            data[self._COMMENT_KEY] = comment
        with open(self._file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self) -> dict:
        """Load and return the data from JSON file."""
        with open(self._file) as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise SerializationException("File does not contain a dictionary.")

            if self._COMMENT_KEY in data:
                del data[self._COMMENT_KEY]
            return data

    def load_comment(self) -> str | None:
        """Load and return the comment from the JSON file."""
        with open(self._file) as f:
            data = json.load(f)
            return data[self._COMMENT_KEY] if self._COMMENT_KEY in data else None
