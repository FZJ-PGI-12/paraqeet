from typing import List, Optional

from cthree.Optimisable import Optimisable


class Generator(Optimisable):
    """
    TODO: Copy as before
    """

    __devices: List

    def __init__(self, devices: Optional[List]):
        self.__devices = devices or []

    def generateSignal(self):
        pass
