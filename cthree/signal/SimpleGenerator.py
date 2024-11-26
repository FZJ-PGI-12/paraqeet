"""Class definition for the Sinusoidal generator model."""

from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import Array, jit, vmap

from cthree.Quantity import Quantity
from cthree.signal.Waveform import Waveform, LocalOscillator
from cthree.signal.Generator import Generator


class CosGenerator(Generator):
    """Simple sinusoidal signal generation.

    Parameters
    ----------
    envelopes : List[cthree.signal.Waveform]
        List of input devices.

    """

    __envs: list[Waveform]
    __phase: Quantity
    _optimisableParameters: list[Quantity] = []

    def __init__(
        self,
        envelopes: list[Waveform] | None,
        frequency: Quantity | None = None,
        phase: Quantity | None = None,
    ):
        self.__envs = envelopes or []

        self.__lo = LocalOscillator(frequency=frequency)

        self.__phase = phase or Quantity(
            np.array(0.0),
            min_value=np.array(-np.pi),
            max_value=np.array(np.pi),
            unit="rad",
            name="Phase",
        )

    def getParameters(self) -> list[Quantity]:
        """Return a list of parameters.

        Collects and returns a list of parameters from the tone, generator
        and the carrier signal.

        Returns
        -------
        List[Quantity]
            All Parameters describing the signal.
        """
        pars = []
        for env in self.__envs:
            pars += env.getParameters()
        pars += self.__lo.getParameters()
        pars += [self.__phase]
        return pars

    def setOptimisableParameters(self, params: list[Quantity]) -> None:
        """Set specified parameters to be optimised.

        Parameters
        ----------
        params : list[Quantity]
        """
        super().setOptimisableParameters(params)

        for dev in self.__envs:
            dev.setOptimisableParameters(params)

        self.__lo.setOptimisableParameters(params)

    def __complexSignal(self, t: np.ndarray) -> Array:
        """Generate a signal for time(s) 't'.

        Doesnt take real value now for ease of gradient computation.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.Array
            Returns the signal vector.

        """
        env = jnp.zeros_like(t)
        for dev in self.__envs:
            env += jnp.reshape(dev.computeOutput(t), env.shape)
        sig = env.conj() * self.__lo.computeOutput(t)
        sig = sig * jnp.exp(-1j * self.__phase.getValue())
        return sig

    def generateSignal(self, t: np.ndarray) -> Array:
        """Generate a signal for time(s) 't'.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.Array
            Returns the signal vector.

        """
        return jnp.real(self.__complexSignal(t))

    def generateSignalGradient(self, t) -> Array:
        """Collect and returns the gradients from all devices.

        Since the signal = Re(env(t).conj() * e^(i*freq*t) * exp(-i*phase))
        Derivative of the signal wrt optimisable parameter of envelope would be
        0.5 * Re(denv(t).conj() * e^(i*freq*t) * e^(-i*phase))
        (TODO - Check the envelope derivatives)

        And derivative of signal wrt parameter of LO would be
        0.5 * Re(env(t).conj() * i*t*e^(i*freq*t) * e^(-i*phase))

        And derivative of signal wrt phase would be
        -0.5 * i * Re(env(t).conj() * e^(i*freq*t) * e^(-i*phase))

        The 0.5 are due to the Wirtinger derivatives due to Re part.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.Array
            Returns the signal gradient vector.

        """
        phase_fac = jnp.exp(-1j * self.__phase.getValue())
        lo_out = self.__lo.computeOutput(t)
        sig = self.__complexSignal(t)
        gradients = jnp.zeros(shape=(t.shape[0], 0))

        # Collect gradients for envelopes
        for dev in self.__envs:
            grad = dev.computeGradient(t).conj()
            if grad.size != 0:
                grad *= jnp.expand_dims(lo_out * phase_fac, axis=1)
            gradients = jnp.append(gradients, 0.5 * jnp.real(grad), axis=1)

        # Collect LO gradients
        lo_freq = self.__lo.getParameters()[0]
        if self._isOptimised(lo_freq):
            gradients = jnp.append(
                gradients,
                jnp.expand_dims(0.5j * t * sig * lo_freq.getScale(), 1),
                axis=1,
            )

        # Collect gradient of Phase
        if self._isOptimised(self.__phase):
            gradients = jnp.append(
                gradients,
                jnp.expand_dims(-0.5j * sig * self.__phase.getScale(), 1),
                axis=1,
            )
        return gradients

    def generateSignalGradientOneTime(self, t) -> Array:
        """Return the gradients from all devices at the given time.

        Since the signal = Re(env(t).conj() * e^(i*freq*t) * exp(-i*phase))
        Derivative of the signal wrt optimisable parameter of envelope would be
        0.5 * Re(denv(t).conj() * e^(i*freq*t) * e^(-i*phase))
        (TODO - Check the envelope derivatives)

        And derivative of signal wrt parameter of LO would be
        0.5*i*t * env(t).conj() * e^(i*freq*t) * e^(-i*phase)

        And derivative of signal wrt phase would be
        -0.5*i * env(t).conj() * e^(i*freq*t) * e^(-i*phase)

        The 0.5 are due to the Wirtinger derivatives due to Re part.

        Parameters
        ----------
        t : float
            Single timestamp.

        Returns
        -------
        jax.Array
            Return the gradients from all devices at one time.

        """
        phase_fac = jnp.exp(-1j * self.__phase.getValue())
        lo_out = jnp.squeeze(self.__lo.computeOutput(t), axis=0)
        sig = self.__complexSignal(t)
        gradients = jnp.zeros(shape=(0,))

        # Collect gradients for envelopes
        for dev in self.__envs:
            grad = jnp.squeeze(dev.computeGradient(t).conj(), axis=0)
            if grad.size != 0:
                grad *= lo_out * phase_fac
            gradients = jnp.append(gradients, 0.5 * jnp.real(grad), axis=0)

        # Collect LO gradients
        lo_freq = self.__lo.getParameters()[0]
        if self._isOptimised(lo_freq):
            gradients = jnp.append(
                gradients, 0.5j * t * sig * lo_freq.getScale(), axis=0
            )

        # Collect gradient of Phase
        if self._isOptimised(self.__phase):
            gradients = jnp.append(
                gradients,
                -0.5j * sig * self.__phase.getScale(),
                axis=0,
            )
        return gradients


class PWCGenerator(Generator):
    """Convert a complex envelope to PWC pulse.

    This sets the pulse parameters to the `tlist` points.
    The gradient of the pulse wrt the PWC bins is 1 at that time point and zero
    everywhere else.

    This Generator doesn't add the LO signal to the envelope pulse.
    Driving with a PWC pulse (without the LO) should be done in the rotating
    frame of drive.

    __envs: list[Waveform]
        List of Envelopes
    __tlist: np.ndarray
        Time grid discritization points

    Parameters
    ----------
    envelopes : List[cthree.signal.Waveform]
        List of input devices.

    """

    __envs: list[Waveform]
    __tlist: np.ndarray
    __inphase: Quantity
    __quadrature: Quantity
    _optimisableParameters: list[Quantity] = []

    def __init__(
        self,
        envelopes: list[Waveform] | None,
        tlist: np.ndarray,
    ):
        self.__envs = envelopes or []
        self.__tlist = tlist
        self.__setInphaseAndQuadrature()

    @property
    def tlist(self) -> np.ndarray:
        """Get time grid discritization for generating PWC pulse.

        Returns
        -------
        np.ndarray
            Array of time points at which envelope is discritized.
        """
        return self.__tlist

    @tlist.setter
    def tlist(self, tlist: np.ndarray) -> None:
        """Set time grid discritization for generating PWC pulse.

        Parameters
        ----------
        tlist : np.ndarray
            Array of time points at which envelope is discritized.
        """
        self.__tlist = tlist
        self.__setInphaseAndQuadrature()

    def __setInphaseAndQuadrature(self) -> None:
        """Generate Inphase and Quadrature Quantities using tlist."""
        env = jnp.zeros_like(self.__tlist)
        for dev in self.__envs:
            env += dev.computeOutput(self.__tlist)

        max_abs = jnp.max(jnp.abs(env))

        self.__inphase = Quantity(
            jnp.real(env),
            min_value=-2 * max_abs,
            max_value=2 * max_abs,
            unit="Hz",
            name="Inphase",
        )
        self.__quadrature = Quantity(
            jnp.imag(env),
            min_value=-2 * max_abs,
            max_value=2 * max_abs,
            unit="Hz",
            name="Quadrature",
        )

    def getParameters(self) -> list[Quantity]:
        """Return a list of parameters.

        Return the inphase and quadrature as parameters.

        Returns
        -------
        List[Quantity]
            All Parameters describing the signal.
        """
        return [self.__inphase, self.__quadrature]

    def setOptimisableParameters(self, params: list[Quantity]) -> None:
        """Set specified parameters to be optimised.

        Optimisable paramters can be inphase and quadrature.

        Parameters
        ----------
        params : list[Quantity]
        """
        super().setOptimisableParameters(params)

    @partial(jit, static_argnums=(0,))
    def __PWCSignal(
        self,
        inphase: np.ndarray,
        quadrature: np.ndarray,
        tlist: np.ndarray,
        t: float,
    ) -> Array:
        """Generate a signal for a single time point 't'.

        The PWC signal is generated by finiding the closest time point
        and returning the correspoinding amplitude value.

        Parameters
        ----------
        inphase: np.ndarray
            1-D vector of step values of real part of the PWC signal.
        quadrature: np.ndarray
            1-D vector of step values of complex part of the PWC signal.
        tlist: np.ndarray
            Time bins of the PWC pulse.
        t : float
            One time point.

        Returns
        -------
        jax.Array
            Returns the PWC signal value at t.

        """
        index = jnp.argmin(jnp.abs(tlist - t))
        return inphase[index] + 1j * quadrature[index]

    def generateSignal(self, t: np.ndarray) -> Array:
        """Generate the PWC signal for time(s) 't'.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.Array
            Returns the signal vector.

        """
        t = jnp.array(t, ndmin=1)
        inphase = self.__inphase.getValue()
        quadrature = self.__quadrature.getValue()
        tlist = self.__tlist
        return jnp.squeeze(
            vmap(self.__PWCSignal, in_axes=(None, None, None, 0))(
                inphase, quadrature, tlist, t
            )
        )

    def generateSignalGradient(self, t: np.ndarray) -> Array:
        """Return signal gradient wrt inphase and quadrature.

        This returns a list of ones as the gradient of the envelope wrt a step
        is 1 for that time bin and 0 everywhere else.

        Parameters
        ----------
        t : np.ndarray
            Array of time steps.

        Returns
        -------
        Array
            PWC signal gradients.
        """
        t = jnp.array(t, ndmin=1)

        grads = []

        inphase_scale = self.__inphase.getScale()
        quadrature_scale = self.__quadrature.getScale()

        if self._isOptimised(self.__inphase):
            grads.append(jnp.ones_like(t) * inphase_scale)
        if self._isOptimised(self.__quadrature):
            grads.append(jnp.ones_like(t) * quadrature_scale)

        return (
            jnp.stack(grads, axis=1)
            if len(grads) > 0
            else jnp.empty((t.shape[0], 0))
        )

    def generateSignalGradientOneTime(self, t: float) -> Array:
        """Return signal gradient wrt inphase and quadrature.

        This returns a list of ones as the gradient of the envelope wrt a step
        is 1 for that time bin and 0 everywhere else.

        Parameters
        ----------
        t : float
            One time step.

        Returns
        -------
        Array
            PWC signal gradients.
        """
        grads = []

        inphase_scale = self.__inphase.getScale()
        quadrature_scale = self.__quadrature.getScale()

        if self._isOptimised(self.__inphase):
            grads.append(inphase_scale)
        if self._isOptimised(self.__quadrature):
            grads.append(quadrature_scale)

        return jnp.stack(grads, axis=0) if len(grads) > 0 else jnp.empty((0,))
