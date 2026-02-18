from abc import ABC, abstractmethod


class Solvable(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_value(self, times):
        pass

    @abstractmethod
    def get_value_and_gradient(self, times):
        pass

    @abstractmethod
    def get_gradient_at_timestep(self, timestep):
        pass
