class Measurement:
    """
    Represents any observable and the process of measurement itself. The observable is measured after the propagation
    class has solved the equation of motion.
    """

    def __init__(self):
        pass

    def measure(self) -> float:
        """
        Measures the observable and returns the value. This function must be implemented by subclasses.
        """
        pass
