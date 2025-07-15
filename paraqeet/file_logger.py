"""Class definition of the file logger object."""

import json
import os

from paraqeet.logger import Logger
from paraqeet.quantity import Quantity


class FileLogger(Logger):
    """Logger that writes messages to a file.

    Parameters
    ----------
    logdir : str, default="."
        Destination directory to store the logs.

    """

    __logdir: str
    __logfile: str
    __resultFile: str

    def __init__(self, logdir: str = ".") -> None:
        self.logdir = logdir

    @property
    def logdir(self) -> str:
        """Returns the current log directory."""
        return self.__logdir

    @logdir.setter
    def logdir(self, logdir: str):
        """Set the destination log directory.

        Stores both the log and the result files.

        Parameters
        ----------
        logdir : str
            Destination directory to store the logs.

        """
        self.__logdir = logdir
        self.__logfile = os.path.join(self.__logdir, "opt.log")
        self.__resultFile = os.path.join(self.__logdir, "opt.result")
        if not os.path.isdir(self.__logdir):
            os.makedirs(self.__logdir)

    def start(self):
        """Start logging."""
        super().start()

    def log(self, params: list[Quantity], infidelity: float):
        """Write the formatted parameters and the goal to the log file.

        Parameters
        ----------
        params : List[Logger]
            List of parameters to be written to the log file.
        infidelity : float
            Goal value to be written to the log file.

        """
        super().log(params, infidelity)
        formattedParams = [param.get_value().tolist() for param in params]
        status = {
            "Eval": self._counter,
            "Parameters": formattedParams,
            "Goal": infidelity,
        }
        with open(self.__logfile, "a") as log:
            log.write(json.dumps(status))
            log.write("\n")
            log.flush()

    def stop(self, result_message: str | None = None):
        """Stop logging and end the log file with the run information.

        Parameters
        ----------
        result_message : str, optional
            The message that the user wants to write at the end of the log file.

        """
        super().stop()
        with open(self.__resultFile, "a") as log:
            if result_message:
                log.write(result_message)
                log.write("\n")
            log.write(f"Finished at {self._stop_time}\n")
            log.write(f"Total runtime: {self._stop_time - self._start_time}")
            log.write("\n")
            log.flush()
