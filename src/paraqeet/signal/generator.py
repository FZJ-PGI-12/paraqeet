"""Class definition of the Generator model."""

from abc import abstractmethod

from paraqeet.differentiable import Differentiable
from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Array


class Generator(Optimizable, Differentiable):
    """Signal generation stack.

    Contrary to most quantum simulators, paraqeet includes a detailed simulation
    of the control stack. Each component in the stack and its functions are
    simulated individually and combined here.

    Example: A local oscillator and arbitrary waveform generator signal
    are put through via a mixer device to produce an effective modulated signal.

    """

    @abstractmethod
    def get_value(self, times: Array) -> Array:
        """Return array with scalar signal value for each time step.

        Parameters
        ----------
        times: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the scalar signal vector.

        """
        pass

    @abstractmethod
    def get_gradient_at_timestep(self, time: Array) -> Array:
        """Return array with the gradient of the signal value for one time step.

        The result has the shape (p,) where 'p' is the parameter index.

        Parameters
        ----------
        time: Array
            One time stamp.

        Returns
        -------
        Array

        """
        pass
