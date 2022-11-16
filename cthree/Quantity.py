import numpy as np

class Quantity:
    """
    TODO: copy from C3 code
    """
    __value = np.array(0.0)

    def __init__(self, value):
        self.__value = value
    
    def get_value(self):
        return self.__value

    def set_value(self, value):
        self.__value = value
