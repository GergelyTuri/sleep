"""Filter functions for signals. Functions in this module should take as their
first argument an array-like object of size (n_ROIs, n_timepoints), and should
return results in an identical format. If the input is a DataFrame, this
should be preserved (along with its index)."""


import pandas as pd

try:
    from src.config import MAXMIN_WINDOW, MAXMIN_SIGMA, QUANTILE_BASELINE, FILTER_MIN_PERIODS
except ImportError:
    from config import MAXMIN_WINDOW, MAXMIN_SIGMA, QUANTILE_BASELINE, FILTER_MIN_PERIODS


def maxmin_filter(signal, window=MAXMIN_WINDOW, sigma=MAXMIN_SIGMA, min_periods=FILTER_MIN_PERIODS):
    """Calculate baseline as the rolling maximum of the rolling minimum of the
    smoothed trace

    Parameters
    ----------
    signal : array, size (n_ROIs, n_timepoints)
    window : int
        Optional, size of the rolling window for max/min/smoothing
    sigma : int
        Standard deviation of the gaussian smoothing kernel
    min_periods : float [0, 1]
        Optional, percentage of values in each window that must be non-NaN to
        return a non-NaN value in the baseline
    """

    kwargs = {
        "window": window,
        "min_periods": int(window * min_periods),
        "center": True,
        "axis": 1,
    }

    smooth_signal = (
        pd.DataFrame(signal).rolling(win_type="gaussian", **kwargs).mean(std=sigma)
    )

    return smooth_signal.rolling(**kwargs).min().rolling(**kwargs).max()


# DEAD CODE? std="sigma" (string literal) should be std=sigma (variable) — crashes at runtime.
# def smooth_quantile_filter(signal, quantile=QUANTILE_BASELINE, window=MAXMIN_WINDOW, sigma=MAXMIN_SIGMA, min_periods=FILTER_MIN_PERIODS):
#     """Estimate baseline as the rolling quantile of the Gaussian-smoothed trace."""
#     kwargs = {
#         "window": window,
#         "min_periods": int(window * min_periods),
#         "center": True,
#         "axis": 1,
#     }
#
#     smooth_signal = (
#         pd.DataFrame(signal).rolling(win_type="gaussian", **kwargs).mean(std=sigma)
#     )
#
#     return smooth_signal.rolling(**kwargs).quantile(quantile)
