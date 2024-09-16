"""Class definition of the Logger model."""

from datetime import datetime
import numpy as np

from cthree.Quantity import Quantity


class Logger:
    """Abstract base class that can be used as a callback in the optimiser."""

    _startTime: datetime
    _stopTime: datetime
    _counter: int

    def start(self):
        """Start logging and set starting values to the run parameters."""
        self._startTime = datetime.now()
        self._counter = 0

    def log(self, params: list[Quantity], infid: np.ndarray):
        """Template function to direct what happens at each log call.

        Parameters
        ----------
        params : List[cthree.Quantity]
            List of parameters to be logged.
        infid : numpy.ndarray
            Goal value to be logged.

        """
        self._counter += 1

    def stop(self, resultMessage: str = None):
        """Template function to stop logging and set end of log parameters.

        Parameters
        ----------
        resultMessage : str, optional
            The message that the user wants to write at the end of the log file.

        """
        self._stopTime = datetime.now()
