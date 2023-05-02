import numpy as np
from cthree.signal.SimpleGenerator import CosGenerator
from cthree.signal.Device import CosTone

LEN_SIG = 1001

ts = np.linspace(0, 10e-9, LEN_SIG)

tone = CosTone()
gen = CosGenerator(devices=[tone])


def test_gen() -> None:
    """
    Computes a sample signal and checks vectorized generation.
    """
    sig = gen.generateSignal(ts)
    assert len(sig) == LEN_SIG
