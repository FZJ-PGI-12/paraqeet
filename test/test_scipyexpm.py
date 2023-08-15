from cthree.propagation.ScipyExpm import ScipyExpm


def test_parameters(model):
    propagation = ScipyExpm(model=model, res=3)
    assert propagation.getParameters() == []


def test_resultion(model):
    propagation = ScipyExpm(model=model, res=3)
    propagation.setResolution(532)
    assert propagation.getResolution() == 532
