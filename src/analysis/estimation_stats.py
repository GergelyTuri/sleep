"""Module for calculating effect size and significance using estimation statistics.

Functions:
- ci_difference_unpaired: Cohen's d and p-value for an unpaired t-test.
- ci_difference_paired: Cohen's d and p-value for a paired t-test.
- mean_activity: Mean activity of upregulated/downregulated cells.
"""

from os.path import join

import numpy as np
import pandas as pd
from scipy import stats


def _cohens_d_unpaired(a: np.ndarray, b: np.ndarray) -> float:
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(
        ((n_a - 1) * np.std(a, ddof=1) ** 2 + (n_b - 1) * np.std(b, ddof=1) ** 2)
        / (n_a + n_b - 2)
    )
    return (np.mean(a) - np.mean(b)) / pooled_std


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return np.mean(diff) / np.std(diff, ddof=1)


def ci_difference_unpaired(awake_arr: np.ndarray, nrem_arr: np.ndarray) -> tuple:
    """Cohen's d and significance for an unpaired t-test between two 1-D arrays.

    Returns
    -------
    (difference, significance) : (float, bool)
        Cohen's d and whether p < 0.05 (unpaired Student's t-test).
    """
    if not isinstance(awake_arr, np.ndarray) or not isinstance(nrem_arr, np.ndarray):
        raise TypeError("Inputs must be NumPy arrays.")
    if awake_arr.ndim != 1 or nrem_arr.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays.")

    difference = _cohens_d_unpaired(awake_arr, nrem_arr)
    _, pvalue = stats.ttest_ind(awake_arr, nrem_arr)
    return difference, bool(pvalue < 0.05)


def ci_difference_paired(awake_arr: np.ndarray, nrem_arr: np.ndarray) -> tuple:
    """Cohen's d and significance for a paired t-test between two 1-D arrays.

    Returns
    -------
    (difference, significance) : (float, bool)
        Cohen's d and whether p < 0.05 (paired Student's t-test).
    """
    if not isinstance(awake_arr, np.ndarray) or not isinstance(nrem_arr, np.ndarray):
        raise TypeError("Inputs must be NumPy arrays.")
    if awake_arr.ndim != 1 or nrem_arr.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays.")

    difference = _cohens_d_paired(awake_arr, nrem_arr)
    _, pvalue = stats.ttest_rel(awake_arr, nrem_arr)
    return difference, bool(pvalue < 0.05)


def mean_activity(data: str, direction: str = "Upregulated") -> pd.DataFrame:
    """Mean activity of upregulated/downregulated cells."""
    dfof = pd.read_csv(join(data, "dfof.csv"))
    stat_results = pd.read_csv(join(data, "Significant_DABEST_NREM.csv"))
    upregulated = list(
        stat_results.query("Direction == @direction")["roi_label"].unique()
    )
    upreg_cells = dfof[dfof["roi_label"].isin(upregulated)]
    upreg_cells.set_index("roi_label", drop=True, inplace=True)
    return upreg_cells.mean()
