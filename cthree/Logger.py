from datetime import datetime
from typing import List

from cthree.Quantity import Quantity


class Logger:
    """
    Abstract base class that can be used as a callback in the optimiser.
    """
    _startTime: datetime
    _stopTime: datetime
    _counter: int

    def start(self):
        self._startTime = datetime.now()
        self._counter = 0

    def log(self, params: List[Quantity], infid: float):
        self._counter += 1

    def stop(self, resultMessage: str = None):
        self._stopTime = datetime.now()
