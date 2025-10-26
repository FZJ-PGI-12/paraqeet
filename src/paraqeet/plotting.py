"""A plotting library for ParaQeet."""

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import matplotlib as mpl

import numpy as np

from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array
from paraqeet.signal.waveform import Waveform
from paraqeet.signal.generator import Generator

matplotlib_inline.backend_inline.set_matplotlib_formats("pdf", "svg")

# Specifying the custom plotting fonts
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = "Tex Gyre Pagella"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.size"] = " 10.0"
mpl.rcParams["axes.labelsize"] = " 11.0"
mpl.rcParams["legend.fontsize"] = "10.0"
mpl.rcParams["xtick.labelsize"] = "10.0"
mpl.rcParams["ytick.labelsize"] = "10.0"
mpl.rcParams["axes.linewidth"] = "1.0"
mpl.rcParams["xtick.major.size"] = "3.0"
mpl.rcParams["xtick.major.width"] = "1.0"
mpl.rcParams["ytick.major.size"] = "3.0"
mpl.rcParams["ytick.major.width"] = "1.0"
mpl.rcParams["axes.titlesize"] = "medium"
mpl.rcParams["figure.titlesize"] = "medium"

# Specifying a custom color palette
default_colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
custom_colors = [
    "#18428A",
    "#FF9258",
    "#57B870",
    "#BF433B",
    "#FF00A6",
    "#FFCC33",
    "#008B8B",
    "#9B5DE5",
] + default_colors

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=custom_colors)  # type: ignore


def plot_signal_and_dynamics(
    generator: Generator,
    propagation: Propagation,
    times: Array,
    axes=None,
    state_labels: list[str] | None = None,
    linestyle="-",
    label="",
    alpha=1,
    linewidth=1.5,
):
    """Plot the signal and the correspoding dynamics.

    If fig or ax is provided then ax[0] is used to plot the signal and ax[1] for dynamics.
    This can be used to plot multiple signals and dyanmics on the same plot.
    """
    states = propagation.propagate(times)
    sig = generator.generate_signal(times) / 1e6 / (2 * np.pi)

    if axes is None:
        _, axes = plt.subplots(2, figsize=(4, 5), sharex=True, dpi=100)

    axes[0].plot(
        times / 1e-9,
        np.real(sig),
        label="Re " + label,
        ls=linestyle,
        alpha=alpha,
        linewidth=linewidth,
    )
    axes[0].plot(
        times / 1e-9,
        np.imag(sig),
        label="Imag " + label,
        ls=linestyle,
        alpha=alpha,
        linewidth=linewidth,
    )
    axes[0].legend(loc=1)
    axes[0].set_ylabel("Amplitude \n" + r"[MHz / $2\pi$]")

    axes[1].plot(
        times / 1e-9,
        np.abs(states)[:, :, 0] ** 2,
        ls=linestyle,
        alpha=alpha,
        linewidth=linewidth,
    )
    axes[1].set_ylabel("Population")
    axes[-1].set_xlabel("Time [ns]")
    if state_labels is not None:
        axes[1].legend(state_labels)

    axes[0].grid(True, linestyle=(1, (1, 5)), linewidth=1)
    axes[1].grid(True, linestyle=(1, (1, 5)), linewidth=1)

    return axes


def plot_signal(
    device: Generator | Waveform,
    times: Array,
    axes=None,
    linestyle="-",
    label="",
    alpha=1,
    linewidth=1.5,
):
    """Plot signal from Generator or Envelope."""
    if isinstance(device, Generator):
        sig = device.generate_signal(times) / 1e6 / (2 * np.pi)
    else:
        sig = device.compute_output(times) / 1e6 / (2 * np.pi)

    if axes is None:
        _, axes = plt.subplots(1, figsize=(4, 3), dpi=100)

    axes.plot(
        times / 1e-9,
        np.real(sig),
        label="Re " + label,
        ls=linestyle,
        alpha=alpha,
        linewidth=linewidth,
    )
    axes.plot(
        times / 1e-9,
        np.imag(sig),
        label="Imag " + label,
        ls=linestyle,
        alpha=alpha,
        linewidth=linewidth,
    )
    axes.grid(True, linestyle=(1, (1, 5)), linewidth=1)
    axes.set_ylabel(r"Amplitude [MHz / $2\pi$]")
    axes.set_xlabel("Time [ns]")
    axes.legend()
    return axes
