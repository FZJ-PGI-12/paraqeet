"""Class definition of the Generator model."""

from abc import abstractmethod

from paraqeet.quantity import Array

from paraqeet.optimisable import Optimisable


class Generator(Optimisable):
    """Signal generation stack.

    Contrary to most quantum simulators, C^3 includes a detailed simulation
    of the control stack. Each component in the stack and its functions are
    simulated individually and combined here.

    Example: A local oscillator and arbitrary waveform generator signal
    are put through via a mixer device to produce an effective modulated signal.

    """

    @abstractmethod
    def generate_signal(self, times: Array) -> Array:
        """Return array with scalar signal value for each time step.

        Parameters
        ----------
        times : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the scalar signal vector.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    @abstractmethod
    def generate_signal_gradient(self, times: Array) -> Array:
        """Return array with gradient of signal value for each time step.

        Abstract method.
        The result has the shape (t,p) where 't' is the time and 'p' is
        the parameter index.

        Parameters
        ----------
        times : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the signal gradient vector.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    @abstractmethod
    def generate_signal_gradient_one_time(self, time: Array) -> Array:
        """Return array with the gradient of the signal value for one time step.

        The result has the shape (p,) where 'p' is the parameter index.

        Parameters
        ----------
        time : float
            One time stamp.

        Returns
        -------
        Array

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()
