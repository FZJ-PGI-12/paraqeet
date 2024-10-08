import json

from cthree.serialisation.Serialiser import Serialiser


class JSONFileSerialiser(Serialiser):
    __file: str

    def __init__(self, file: str):
        self.__file = file

    def save(self, data: dict) -> None:
        with open(self.__file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def load(self) -> dict:
        with open(self.__file) as f:
            return json.load(f)
