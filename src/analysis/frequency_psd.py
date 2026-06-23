"""
Methods for calculating the frequency and power spectral density (PSD) of a signal.
"""

import numpy as np
import pandas as pd
from scipy import signal

try:
    from src.config import MAX_AUTOCORR_LAGS
except ImportError:
    from config import MAX_AUTOCORR_LAGS


def freq_calc(data: pd.Series, fs: int = 10, resolution: float = 0.01):
    """
    Calculate the frequency spectrum of the given data using Welch's method.

    Parameters:
    - data: pd.Series
        The input data for frequency analysis.
    - fs: int, optional
        The sampling frequency of the data. Default is 10.
    - resolution: float, optional
        The desired frequency resolution. Default is 0.01.

    Returns:
    - frequencies: array-like
        The frequencies corresponding to the power spectral density.
    - psd: array-like
        The power spectral density of the input data.
    """
    nperseg = int(fs / resolution)
    frequencies, psd = signal.welch(data, fs=fs, nperseg=nperseg,
                                     detrend="linear")
    return frequencies, psd


def calculate_autocorrelations(df):
    """
    Calculate autocorrelations for each row in the DataFrame.

    Parameters:
    - df: pandas DataFrame
        The input DataFrame containing the signals.

    Returns:
    - np.array
        An array of autocorrelations for each row in the DataFrame.
    """
    autocorrelations = []
    for index, row in df.iterrows():
        sig = row.values
        autocorr = np.correlate(sig, sig, mode="full")
        autocorr = autocorr[autocorr.size // 2:]  # Keep second half
        autocorr /= autocorr[0]  # Normalize
        autocorrelations.append(autocorr[:MAX_AUTOCORR_LAGS])
    return np.array(autocorrelations)


def calculate_psd(data, sampling_rate, freq_range=None):
    """
    Calculate the Power Spectral Density (PSD) for a given signal.

    Parameters:
    - data: 2D numpy array of the input signals (signals x time points).
    - sampling_rate: Sampling rate of the data in Hz.
    - freq_range: Optional tuple specifying the frequency range (min_freq, max_freq).

    Returns:
    - freqs: 1D numpy array of frequencies corresponding to the PSD values.
    - psd_values: 2D numpy array of PSD values (signals x PSD values).
    """
    n = data.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    psd_values = []

    for sig in data:
        fft_result = np.fft.rfft(sig)
        psd = np.abs(fft_result) ** 2 / n
        psd_values.append(psd)

    psd_values = np.array(psd_values)

    if freq_range is not None:
        min_freq, max_freq = freq_range
        freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
        freqs = freqs[freq_mask]
        psd_values = psd_values[:, freq_mask]

    return freqs, psd_values
