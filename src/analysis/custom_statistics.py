import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smm
from scipy import stats


def significance_calc(nrem_array: np.array, awake_array: np.array) -> np.array:
    """
    Calculates the significance of the difference between two arrays of
    data using the Mann-Whitney U test.

    Parameters:
    nrem_array (np.array): An array of data representing NREM sleep.
    awake_array (np.array): An array of data representing wakefulness.

    Returns:
    np.array: An array of indices representing the cells that have
             a significant difference between NREM and wakefulness.
    """
    p_values = []

    for nrem_a, awake_a in zip(nrem_array, awake_array):
        stat, p = stats.mannwhitneyu(
            nrem_a, awake_a, alternative="greater"
        )  # alternative='greater' if hypothesis is that NREM values are higher
        p_values.append(p)

    # Multiple testing correction using statsmodels
    reject_list, pvals_corrected, alphacSidak, alphacBonf = smm.multipletests(
        p_values, alpha=0.05, method="bonferroni"
    )
    # reject_list is a boolean array where True means that we reject the null hypothesis for that test
    # pvals_corrected are the adjusted p-values

    # Find indices of significant tests
    significant_cells = np.where(reject_list)[0]
    return significant_cells


def calc_mean_sem_ci(data: pd.DataFrame, state: str, confidence: float = 0.95):
    """Calculate the mean and standard error of the mean (SEM) for each cell"""
    mean = data.loc[state].mean()
    sem = stats.sem(data.loc[state], axis=0, nan_policy="omit")
    ci = stats.t.ppf((1 + confidence) / 2.0, len(data.loc[state]) - 1)
    return mean, sem, ci
