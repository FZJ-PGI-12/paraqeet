"""Class definition of the Logger model."""

from datetime import datetime

from paraqeet.quantity import Quantity


class Logger:
    """Abstract base class that can be used as a callback in the optimiser."""

    _start_time: datetime
    _stop_time: datetime
    _counter: int

    def start(self):
        """Start logging and set starting values to the run parameters."""
        self._start_time = datetime.now()
        self._counter = 0

    def log(self, params: list[Quantity], infidelity: float):
        """Template function to direct what happens at each log call.

        Parameters
        ----------
        params : List[Quantity]
            List of parameters to be logged.
        infid : float
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
