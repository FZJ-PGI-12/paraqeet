from typing import List

from cthree.Optimisable import Optimisable


class Generator(Optimisable):
    """
    TODO: Copy as before
    """
    __devices: List

    def __init__(self, devices: List):
        self.__devices = devices

    def generateSignal(self):
        pass
