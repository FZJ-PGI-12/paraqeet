"""Class definition of the Logger model."""

from jax import Array
from datetime import datetime

from cthree.quantity import Quantity


class Logger:
    """Abstract base class that can be used as a callback in the optimiser."""

    _start_time: datetime
    _stop_time: datetime
    _counter: int

    def start(self):
        """Start logging and set starting values to the run parameters."""
        self._start_time = datetime.now()
        self._counter = 0

    def log(self, params: list[Quantity], infidelity: Array):
        """Template function to direct what happens at each log call.

        Parameters
        ----------
        params : List[cthree.Quantity]
            List of parameters to be logged.
        infid : numpy.ndarray
            Goal value to be logged.

        """
        self._counter += 1

    def stop(self, result_message: str | None = None):
        """Template function to stop logging and set end of log parameters.

        Parameters
        ----------
        result_message : str, optional
            The message that the user wants to write at the end of the log file.

        """
        self._stop_time = datetime.now()
