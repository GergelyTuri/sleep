"""Plotting helpers for PSD and spectral density results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from src.analysis.custom_statistics import calc_mean_sem_ci
except ImportError:
    from analysis.custom_statistics import calc_mean_sem_ci


def plot_psd(
    freqs,
    psd_values,
    ax,
    title="PSD of Autocorrelation",
    xlabel="Frequency (Hz)",
    ylabel="Power Spectral Density",
    label=None,
):
    """
    Plot a line plot of the Power Spectral Density (PSD) values.

    Parameters:
    - freqs: 1D numpy array of frequencies corresponding to the PSD values.
    - psd_values: 2D numpy array of PSD values for each signal (rows for signals, columns for PSD values).
    - ax: matplotlib.axes.Axes object where the plot will be drawn.
    - title: String, title of the plot.
    - xlabel: String, label for the x-axis.
    - ylabel: String, label for the y-axis.
    """
    avg_psd = np.mean(psd_values, axis=0)
    ax.plot(freqs, avg_psd, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax


def spectral_density_plot(
    data: pd.DataFrame,
    states: list,
    ax: plt.Axes = None,
    labels: dict = None,
) -> plt.Axes:
    """Plot the spectral density of two states.

    Parameters:
    data: pd.DataFrame
    states: list of two states
    ax: plt.Axes
    labels: dict of labels
        implemented keys: "suptitle", "title"

    returns: plt.Axes
    """
    if labels is None:
        labels = {}

    if ax is None:
        fig, ax = plt.subplots()
        figure_created = True
    else:
        figure_created = False

    if figure_created:
        fig.suptitle(labels.get("suptitle", "Power Spectral Density: Awake vs NREM"))

    state1_mean, state1_sem, ci = calc_mean_sem_ci(data, states[0])
    state2_mean, state2_sem, _ = calc_mean_sem_ci(data, states[1])

    state1_lower = state1_mean - ci * state1_sem
    state1_upper = state1_mean + ci * state1_sem

    state2_lower = state2_mean - ci * state2_sem
    state2_upper = state2_mean + ci * state2_sem

    x_axis = np.linspace(0, 5, num=len(state1_mean))
    ax.semilogy(x_axis, state1_mean, label=states[0])
    ax.semilogy(x_axis, state2_mean, label=states[1])
    ax.fill_between(x_axis, state1_lower, state1_upper, color="blue", alpha=0.2)
    ax.fill_between(x_axis, state2_lower, state2_upper, color="orange", alpha=0.2)

    ax.legend(loc="lower left")
    ax.set_title(labels.get("title", "Cell"))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power/Frequency (dB/Hz)")
    ax.set_xlim(0.01, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax
