"""Class definition of the Logger model."""

from abc import ABC
from datetime import datetime

from paraqeet.quantity import Quantity


class Logger(ABC):
    """Abstract base class that can be used as a callback in the optimizer."""

    _start_time: datetime
    _stop_time: datetime
    _counter: int

    def start(self):
        """Start logging and set starting values to the run parameters."""
        self._start_time = datetime.now()
        self._counter = 0

    def log(self, params: list[Quantity], infidelity: float):
        """Template function to direct what happens at each log call.

        Args:
            params: List of parameters to be logged.
            infidelity: Goal value to be logged.
        """
        self._counter += 1

    def stop(self, result_message: str | None = None):
        """Template function to stop logging and set end of log parameters.

        Args:
            result_message: The message that the user wants to write at the
                end of the log file.
        """
        self._stop_time = datetime.now()
