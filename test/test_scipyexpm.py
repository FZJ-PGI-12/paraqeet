from cthree.propagation.ScipyExpm import ScipyExpm


def test_parameters(model):
    propagation = ScipyExpm(model=model, res=3)
    assert propagation.getParameters() == []
