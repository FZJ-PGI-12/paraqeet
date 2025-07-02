"""Testing the optimisation map."""

from paraqeet.optimisation_map import OptimisationMap

from paraqeet.signal.iq_mixer import IQMixer
from paraqeet.signal.envelopes import ConstantEnvelope, FlatTopGaussianEnvelope

tone = ConstantEnvelope()
gen = IQMixer(envelopes=[tone])
params = tone.get_parameters()
optmap = OptimisationMap()


def testGetParameters() -> None:
    """Get the test parameters from the optimisation map."""
    optmap.add(tone, params)
    assert len(optmap.get_all_parameters()) == len(params)


def testParametersOverwrite() -> None:
    """Override parameters from the optimisation map."""
    optmap.add(tone, [params[1]])
    assert optmap.get_all_parameters() == [params[1]]


def testFilter() -> None:
    """Test for filtered parameters."""
    tone2 = FlatTopGaussianEnvelope()
    optmap.add(tone2)

    def HzFilter(par):
        return par.get_unit() == "Hz"

    # Manual filtering
    pars = optmap.get_all_parameters()
    filterd = []
    for par in pars:
        if HzFilter(par):
            filterd.append(par)

    # Builtin filter
    optmap.filter_parameters(HzFilter)
    pars = optmap.get_all_parameters()
    assert pars == filterd
