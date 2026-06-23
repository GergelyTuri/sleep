"""Module for calculating effect size and significance using estimation statistics.

This module provides functions for calculating Cohen's d effect size and p-value for unpaired and paired student t-tests
between two sets of numbers. It uses estimation statistics to evaluate the difference between the two sets of numbers.

Functions:
- ci_difference_unpaired: Calculates Cohen's d effect size and p-value for an unpaired student t-test.
- ci_difference_paired: Calculates Cohen's d effect size and p-value for a paired student t-test.
- mean_activity: Calculates the mean activity of upregulated/downregulated cells.

For more information on estimation statistics, refer to the documentation: https://acclab.github.io/DABEST-python-docs/tutorial.html
"""

from os.path import join

import dabest
import numpy as np
import pandas as pd


def ci_difference_unpaired(
    awake_arr: np.array, nrem_arr: np.array, resamples: int = 5000
) -> tuple:
    """

    Calculates the Cohen's d effect size and p-value for an unpaired student t-test between two sets of numbers
    Uses estimation stats to evaluate the difference between two sets of numbers.
    The documentation can be found here: https://acclab.github.io/DABEST-python-docs/tutorial.html

    Parameters:
    ===========
    awake_arr: numpy array
        1-D array of compiled dfof values for awake periods
    nrem_arr: numpy array
        1-D array of compiled dfof values for NREM periods
    resamples: int
        number of resamples to generate the effect size bootstrap

    Returns:
    ========
    A tuple of two floats:
    - difference: The Cohen's d effect size between the two arrays
    - significance: boolean depending on the p-value for the (unpaired) student t-test
    """
    if not isinstance(awake_arr, np.ndarray) or not isinstance(nrem_arr, np.ndarray):
        raise TypeError("Inputs must be NumPy arrays.")
    if awake_arr.ndim != 1 or nrem_arr.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays.")

    dabest_dict = {"awake": awake_arr, "nrem": nrem_arr}
    dabest_df = pd.DataFrame.from_dict(dabest_dict, orient="index").transpose()

    # calculating the stats
    two_groups_unpaired = dabest.load(
        dabest_df, idx=("awake", "nrem"), resamples=resamples
    )
    stats = two_groups_unpaired.cohens_d
    significance = stats.results["pvalue_students_t"][0] < 0.05
    difference = stats.results["difference"][0]

    return difference, significance


def ci_difference_paired(
    awake_arr: np.array, nrem_arr: np.array, resamples: int = 5000
) -> tuple:
    """
    Calculates the Cohen's d effect size and p-value for a paired student t-test between two sets of numbers
    Uses estimation stats to evaluate the difference between two sets of numbers.
    The documentation can be found here: https://acclab.github.io/DABEST-python-docs/tutorial.html

    Parameters:
    ===========
    awake_arr: numpy array
        1-D array of compiled dfof values for awake periods
    nrem_arr: numpy array
        1-D array of compiled dfof values for NREM periods
    resamples: int
        number of resamples to generate the effect size bootstrap

    Returns:
    ========
    A tuple of two floats:
    - difference: The Cohen's d effect size between the two arrays
    - significance: boolean depending p-value for the (paired) student t-test
    """
    if not isinstance(awake_arr, np.ndarray) or not isinstance(nrem_arr, np.ndarray):
        raise TypeError("Inputs must be NumPy arrays.")
    if awake_arr.ndim != 1 or nrem_arr.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays.")

    dabest_dict = {"awake": awake_arr, "nrem": nrem_arr}
    dabest_df = pd.DataFrame.from_dict(dabest_dict, orient="index").transpose()

    # calculating the stats
    two_groups_paired = dabest.load(
        dabest_df, idx=("awake", "nrem"), resamples=resamples
    )
    stats = two_groups_paired.cohens_d
    significance = stats.results["pvalue_students_t"][0] < 0.05
    difference = stats.results["difference"][0]

    return difference, significance


def mean_activity(data: str, direction: str = "Upregulated") -> pd.DataFrame:
    """
    Calculates the mean activity of upregulated/downregulated cells.
    :param data: str, path to the data folder
    :param direction: str, 'Upregulated' or 'Downregulated'
    :return: pd.DataFrame, mean activity of upregulated/downregulated cells
    """
    dfof = pd.read_csv(join(data, "dfof.csv"))
    stat_results = pd.read_csv(join(data, "Significant_DABEST_NREM.csv"))
    upregulated = list(
        stat_results.query("Direction == @direction")["roi_label"].unique()
    )
    upreg_cells = dfof[dfof["roi_label"].isin(upregulated)]
    upreg_cells.set_index("roi_label", drop=True, inplace=True)
    return upreg_cells.mean()
