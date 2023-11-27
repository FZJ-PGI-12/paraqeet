import os
import json

from typing import List

from cthree.Logger import Logger
from cthree.Quantity import Quantity


class FileLogger(Logger):
    """
    Logger that writes messages to a file.
    """
    __logdir: str
    __rundir: str
    __logfile: str
    __resultFile: str

    def __init__(self, logdir: str = ".") -> None:
        self.__logdir = os.path.join(logdir, "optim_logs")
        if not os.path.isdir(self.__logdir):
            os.makedirs(self.__logdir)

    def start(self):
        super().start()
        self.__rundir = str(self._startTime)
        os.makedirs(os.path.join(self.__logdir, self.__rundir))
        self.__logfile = os.path.join(self.__logdir, self.__rundir, "opt.log")
        self.__resultFile = os.path.join(self.__logdir, self.__rundir, "opt.result")

    def log(self, params: List[Quantity], infidelity: float):
        super().log(params, infidelity)
        formattedParams = [param.getValue().tolist() for param in params]
        status = {"Eval": self._counter, "Parameters": formattedParams, "Goal": infidelity}
        with open(self.__logfile, "a") as log:
            log.write(json.dumps(status))
            log.write("\n")
            log.flush()

    def stop(self, resultMessage: str = None):
        super().stop()
        if resultMessage:
            with open(self.__logfile, "a") as log:
                log.write(resultMessage)
                log.write("\n")
        with open(self.__resultFile, "a") as log:
            log.write(f"Finished at {self._stopTime}\n")
            log.write(f"Total runtime: {self._stopTime - self._startTime}")
            log.write("\n")
            log.flush()
