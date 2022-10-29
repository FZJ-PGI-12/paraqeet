from Optimisable import Optimisable
from model.Model import Model


class Propagation(Optimisable):
    _model: Model

    def __init__(self, model: Model):
        self._model = model

    def propagate(self):
        pass
