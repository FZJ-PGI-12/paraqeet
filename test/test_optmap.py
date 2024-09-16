"""Testing the optimisation map."""

from cthree.OptimisationMap import OptimisationMap

from cthree.signal.SimpleGenerator import CosGenerator
from cthree.signal.Device import CosTone, CosToneErf

tone = CosTone()
gen = CosGenerator(devices=[tone])
params = tone.getParameters()
optmap = OptimisationMap()


def testGetParameters() -> None:
    """Get the test parameters from the optimisation map."""
    optmap.add(tone, params)
    assert len(optmap.getAllParameters()) == 3


def testParametersOverwrite() -> None:
    """Override parameters from the optimisation map."""
    optmap.add(tone, [params[1]])
    assert optmap.getAllParameters() == [params[1]]


def testFilter() -> None:
    """Test for filtered parameters."""
    tone2 = CosToneErf()
    optmap.add(tone2)

    def HzFilter(par):
        return par.getUnit() == "Hz"

    # Manual filtering
    pars = optmap.getAllParameters()
    filterd = []
    for par in pars:
        if HzFilter(par):
            filterd.append(par)

    # Builtin filter
    optmap.filterParameters(HzFilter)
    pars = optmap.getAllParameters()
    assert pars == filterd
