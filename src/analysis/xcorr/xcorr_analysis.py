"""
Modul dealing with crosscorrelation analyses.
Author: Gergely Turi
email: gt2253@cumc.columbia.edu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def heat_map(data_frame: pd.DataFrame, ax: plt.axes,
             title: str, vmin: float,
             vmax: float)-> plt.axes:
    """
    Plots a customized heatmap of a correlation matrix.
    
    Parameters:
    -----------

    df: pd.DataFrame
    ax: plt.axes
    title: str
    vmin, vmax: float (min and max values for the colorbar)

    Returns:
    --------
    plt.axes

    """
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    mask= np.zeros_like(data_frame)
    mask[np.triu_indices_from(mask)] = True
    
    sns.heatmap(data_frame, cmap=cmap, center=0, square=True, linewidths=.5,
              annot=True, fmt='.2f', cbar_kws={"shrink": .5}, ax=ax, mask=mask,
              vmax=vmax, vmin=vmin, cbar=False)
    
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')