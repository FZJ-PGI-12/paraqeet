# Helper functions that are common tests for all propagation implementations
import numpy as np

from paraqeet.propagation.diffrax_ode import DiffraxODE
from paraqeet.propagation.euler import Euler
from paraqeet.propagation.expm import Expm
from paraqeet.propagation.expm_chebyshev import ExpmChebyshev
from paraqeet.propagation.propagation import Propagation
from paraqeet.propagation.utils import schrodinger_step
from paraqeet.propagation.vern7 import Vern7


def check_propagation(propagation: Propagation, dimension: int):
    random_time_vector = np.linspace(0.0, np.random.randint(1, 10) * np.random.rand(), np.random.randint(2, 10))
    state = np.random.random(dimension) + 1j * np.random.random(dimension)
    state = state / np.sqrt(np.vdot(state, state))
    state = np.expand_dims(state, axis=1)
    propagation.initial_state = state
    propagation.get_value(random_time_vector)


def make_propagation(method, eom_func, resolution, initial_state, step_function=schrodinger_step):
    """Return the propagation method of the given name for one equation of motion.

    Lets the tests of the gradient methods run over every propagation without spelling out the
    constructor of each of them.
    """
    if method == "expm":
        return Expm(eom_func=eom_func, resolution=resolution, initial_state=initial_state)
    if method == "chebyshev":
        return ExpmChebyshev(eom_func=eom_func, resolution=resolution, initial_state=initial_state)
    if method == "euler":
        return Euler(eom_func=eom_func, resolution=resolution, initial_state=initial_state)
    if method == "vern7":
        return Vern7(
            eom_func=eom_func,
            resolution=resolution,
            initial_state=initial_state,
            step_function=step_function,
        )
    if method == "diffrax":
        return DiffraxODE(
            eom_func=eom_func,
            resolution=resolution,
            initial_state=initial_state,
            step_function=step_function,
            samples_per_step=2,
        )
    raise ValueError(f"Unknown propagation method '{method}'.")
