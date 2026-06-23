""" Module with functions for clustering and dimensionality reduction.
Author: Gergely Turi
"""

from os.path import join
from typing import Optional, Union

try:
    from src.config import MIN_SLEEP_EPOCH_FRAMES
except ImportError:
    from config import MIN_SLEEP_EPOCH_FRAMES

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform


def df_generator(data):
    """
    Generate DataFrame based on sleep intervals from the given data.

    Args:
        data (pd.DataFrame): The input data containing variables suggesting sleep or NREM state.

    Returns:
        pd.DataFrame: DataFrame containing sleep intervals (start index, end index, and interval length).
    """
    # Assuming data is your original DataFrame containing 'NREM' and 'awake' columns
    data['indicator'] = data.apply(lambda row: 1 if row['NREM'] else 0 if row['awake'] else None, axis=1)

    # Initialize an empty list to store dictionaries
    df_data = []

    # Your existing code for data processing goes here
    index = [1] + [i+1 for i in range(len(data['indicator'])-1) if data['indicator'][i] != data['indicator'][i+1]]
    for i in range(len(index)-1):
        start = index[i]
        end = index[i+1] - 1
        df_data.append({'n': i, 'sleep': data.iloc[start, -1], 'length': end - start + 2})

    # Create DataFrame from list of dictionaries
    df = pd.DataFrame(df_data)
    df['length'] = pd.to_numeric(df['length'])

    # Filter out rows where length < 600
    del_rows = df[df['length'] < MIN_SLEEP_EPOCH_FRAMES]['n'].tolist()
    df_sleep = df[~df['n'].isin(del_rows)].reset_index(drop=True)

    # Update start and end columns
    df_sleep['n'] = df_sleep.index
    df_sleep['end'] = df_sleep['length'].cumsum() - 1
    df_sleep['start'] = df_sleep['end'].shift(1) + 1
    df_sleep.loc[0, 'start'] = 0

    # Convert columns to integers
    df_sleep['sleep'] = df_sleep['sleep'].astype(int)
    df_sleep['start'] = df_sleep['start'].astype(int)
    df_sleep['length'] = df_sleep['length'].astype(int)
    df_sleep['end'] = df_sleep['end'].astype(int)
    return df_sleep
    
def process_dfof_mc(dfof: pd.DataFrame, summary_sleep: pd.DataFrame) -> dict:
    """
    Process dF/F data based on awake and sleep intervals.

    Args:
        dfof (pd.DataFrame): The dF/F data.
        summary_sleep (pd.DataFrame): The summary of awake and sleep intervals.

    Returns:
        dict: A dictionary containing the processed dF/F data for awake and sleep intervals.
    """
    awake = []
    sleep = []
    for index, row in summary_sleep.iterrows():
        if row['sleep'] == 0:
            awake.extend(range(row['start'], row['end'] + 1))
        else:
            sleep.extend(range(row['start'], row['end'] + 1))
    
    d_awake = dfof.iloc[:, awake] 
    d_sleep = dfof.iloc[:, sleep]
    return {'d_awake': d_awake, 'd_sleep': d_sleep}

def interval_length_calculator(data: pd.DataFrame,
                                state_column: str,
                                  state_value: int) -> pd.DataFrame:
    """
    Calculates the length, start, and stop indices of intervals for a
      specific state.

    Args:
        data (pd.DataFrame): The DataFrame containing interval data.
        state_column (str): Column name indicating the state for each interval.
        state_value (int): The value of the state to filter intervals by.

    Returns:
        pd.DataFrame: A DataFrame with columns for the interval number ('n'),
        the start index, the stop index, and the length of each interval of 
        the specified state.
    """
    # Ensure data is not altered outside the function
    data = data.copy()

    # Convert the state column to ensure compatibility
    data[state_column] = data[state_column].astype(int)
    
    # Detect changes to and from the target state
    is_target_state = data[state_column] == state_value
    starts = is_target_state & (~is_target_state.shift(fill_value=False))
    stops = (~is_target_state) & is_target_state.shift(fill_value=False)

    # Prepare for data accumulation
    rows = []

    # Iterate over starts
    for start in data[starts].index:
        # Find the corresponding stop
        stop = data[stops & (data.index > start)].index.min()
        # If there's no corresponding stop, use the last index
        if not pd.isna(stop):
            stop -= 1 
        else:
            stop = data.index[-1]
        
        length = stop - start +1
        rows.append({'start': start, 'stop': stop, 'length': length})
    
    # Create DataFrame from accumulated rows
    result = pd.DataFrame(rows)
    
    # Add interval numbers
    result.insert(0, 'n', range(1, len(result) + 1))
    
    return result

def process_dfof_intervals(dfof: Union[pd.DataFrame, pd.Series], interval_df: pd.DataFrame):
    """
    Process dF/F data based on specified intervals.

    Args:
        dfof (pd.DataFrame): The dF/F data.
        interval_df (pd.DataFrame): The DataFrame with intervals, containing 'start' and 'stop' columns.

    Returns:
        pd.DataFrame: A DataFrame containing the dF/F data filtered by the specified intervals.
    """
    filtered_dfof = pd.DataFrame()
    
    # Ensure dfof is a DataFrame, even if a Series is provided
    if isinstance(dfof, pd.Series):
        # Convert the Series to a DataFrame with one row and multiple columns
        dfof = pd.DataFrame([dfof.values], columns=dfof.index)

    # Iterate over each interval
    for _, row in interval_df.iterrows():
        start = row['start']
        stop = row['stop']
        
        # Select the dF/F data for the current interval and append it
        current_interval_data = dfof.iloc[:, start:stop+1]  # Including stop index
        filtered_dfof = pd.concat([filtered_dfof, current_interval_data], axis=1)
    
    return filtered_dfof

def calculate_cosine_distance(data: pd.DataFrame, state: str, save_path: Optional[str] = None,
                              normalize: bool = True) -> pd.DataFrame:
    """
    Calculates the cosine distance matrix for the given data.

    Parameters:
    data (pd.DataFrame): The input data.
    state (str): The brain state of the data.
    save_path (Optional[str]): Path to save the distance matrix. If provided, the matrix is saved to this path.

    Returns:
    pd.DataFrame: The cosine distance matrix.
    """
    X = data.values
    # Calculate pairwise cosine distances and convert to square form
    dist = squareform(pdist(X, metric='cosine'))

    # Normalize the distance matrix
    if normalize:
        min_val, max_val = np.nanmin(dist), np.nanmax(dist)
        dist_data = (dist - min_val) / (max_val - min_val)
    else:
        dist_data = dist

    if save_path:
        pd.DataFrame(dist_data).to_csv(join(save_path, f'ts_dist_{state}.csv'), index=False)

    return pd.DataFrame(dist_data)

def sort_distance_matrix(dist_matrix):
    """Reorder rows and columns of a square distance matrix by hierarchical clustering."""
    # Convert the cosine distance matrix to a condensed form since linkage
    # expects a condensed distance matrix
    condensed_dist = squareform(dist_matrix, checks=False)
    
    # Perform hierarchical/agglomerative clustering
    linkage_matrix = linkage(condensed_dist, method='average')
    
    # Get the order of rows and columns according to the clustering
    sorted_indices = leaves_list(linkage_matrix)
    
    # Sort the distance matrix according to the clustering result
    sorted_dist_matrix = dist_matrix.iloc[sorted_indices, :].iloc[:, sorted_indices]
    
    return sorted_dist_matrix
