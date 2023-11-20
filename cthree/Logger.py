import os
import datetime
import json

from typing import List


class Logger:
    "Basic class to log optimisations"
    __logdir: str
    __rundir: str
    __counter: int

    def __init__(self, logdir: str = ".") -> None:
        self.__logdir = os.path.join(logdir, "optim_logs")
        if not os.path.isdir(self.__logdir):
            os.makedirs(self.__logdir)

    def start(self):
        self.__start_time = datetime.datetime.now()
        self.__rundir = str(self.__start_time)
        os.makedirs(os.path.join(self.__logdir, self.__rundir))
        self.__logfile = os.path.join(self.__logdir, self.__rundir, "opt.log")
        self.__counter = 0

    def write_msg(self, status: str):
        with open(self.__logfile, "a") as log:
            log.write(status)
            log.write("\n")
            log.flush()

    def write_json(self, params: List, infid: float):
        self.__counter += 1
        status = {"Eval": self.__counter, "Parameters": params, "Goal": infid}
        with open(self.__logfile, "a") as log:
            log.write(json.dumps(status))
            log.write("\n")
            log.flush()

    def stop(self):
        with open(self.__logfile, "a") as log:
            stop_time = datetime.datetime.now()
            log.write(f"Finished at {stop_time}\n")
            log.write(f"Total runtime: {stop_time - self.__start_time}")
            log.write("\n")
            log.flush()
