"""Class definition of the Generator model."""

from abc import abstractmethod

import numpy as np

from cthree.Optimisable import Optimisable
from cthree.signal.Device import Device


class Generator(Optimisable):
    """Signal generation stack.

    Contrary to most quantum simulators, C^3 includes a detailed simulation
    of the control stack. Each component in the stack and its functions are
    simulated individually and combined here.

    Example: A local oscillator and arbitrary waveform generator signal
    are put through via a mixer device to produce an effective modulated signal.

    """

    __chains: dict[str, dict[str, list[str]]] = {}
    __devices: list[Device]

    @abstractmethod
    def generateSignal(self, times: np.ndarray) -> np.ndarray:
        """Return array with scalar signal value for each time step.

        Parameters
        ----------
        times : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Returns the scalar signal vector.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    @abstractmethod
    def generateSignalGradient(self, times: np.ndarray) -> np.ndarray:
        """Return array with gradient of signal value for each time step.

        Abstract method.
        The result has the shape (t,p) where 't' is the time and 'p' is
        the parameter index.

        Parameters
        ----------
        times : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Returns the signal gradient vector.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    @abstractmethod
    def generateSignalGradientOneTime(self, time: float) -> np.ndarray:
        """Return array with the gradient of the signal value for one time step.

        The result has the shape (p,) where 'p' is the parameter index.

        Parameters
        ----------
        time : float
            One time stamp.

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()
