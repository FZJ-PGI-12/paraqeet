import numpy as np
from cthree.signals.SimpleGenerator import CosGenerator

LEN_SIG = 1001

ts = np.linspace(0, 10e-9, LEN_SIG)
gen = CosGenerator()


def test_gen() -> None:
    """
    Computes a sample signal and checks vectorized generation.
    """
    sig = gen.generateSignal(ts)
    assert len(sig) == LEN_SIG
