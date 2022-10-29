from typing import List

from Optimisable import Optimisable


class Generator(Optimisable):
    """
    Copy as before
    """
    __devices: List

    def __init__(self, devices: List):
        self.__devices = devices

    def generateSignal(self):
        pass
