import pytest
import numpy as np
from cthree.signal.SimpleGenerator import CosGenerator
from cthree.signal.Device import CosTone

LEN_SIG = 1001


@pytest.fixture
def time_samples():
    return np.linspace(0, 10e-9, LEN_SIG)


@pytest.fixture
def gen():
    tone = CosTone()
    return CosGenerator(devices=[tone])


def test_gen(gen, time_samples) -> None:
    """
    Computes a sample signal and checks vectorized generation.
    """
    sig = gen.generateSignal(time_samples)
    assert len(sig) == LEN_SIG
