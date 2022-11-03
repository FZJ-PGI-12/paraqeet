from cthree.Optimisable import Optimisable
from cthree.model.Model import Model


class Propagation(Optimisable):
    """
    Abstract base class for any implementation that can solve the equation of motion. The right-hand side of the
    equation is provided by the underlying model.
    """
    _model: Model

    def __init__(self, model: Model):
        self._model = model

    def propagate(self):
        pass
