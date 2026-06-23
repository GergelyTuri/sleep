"""
Module for miscallenious calculations and helper functions.
Needs to be orgainzed.
Author: Gergely Turi
email: gt2253@cumc.columbia.edu
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import scipy.stats as stats
from collections import Counter

try:
    from src.config import MIN_EPOCH_FRAMES, IMMOBILITY_VELOCITY_THRESHOLD
except ImportError:
    from config import MIN_EPOCH_FRAMES, IMMOBILITY_VELOCITY_THRESHOLD

def label_consecutive_states(data: pd.Series, state: str='NREM') -> pd.Series:

    """    
    Labels consecutive occurrences of a particular state in a Pandas Series.
    
    Parameters:
    -----------
    data : pandas.Series
        The Series containing booleans referring to the state to be labeled.
    state : str, optional (default='NREM')
        The state to label. At least 100 consecutive occurrences of this state
        will be marked with a label.
        
    Returns:
    --------
    pandas.Series
        A Series with a new index indicating consecutive occurrences of the
        specified state.
        
    Example:
    --------
    >>> data = pd.Series([True, True, True, False, True, True, False, True])
    >>> label_consecutive_states(data, 'NREM')
    0       False
    1    NREM1
    2    NREM1
    3       False
    4    NREM2
    5    NREM2
    6       False
    7    NREM3
    dtype: object

    """

    df = pd.Series(False, name=f'{state}_label', index=data.index)

    consecutive_count = 0
    label_count = 1

    for i, val in data.items():
        if val:
            consecutive_count += 1
            if consecutive_count > MIN_EPOCH_FRAMES:
                df.loc[i-consecutive_count+1:i] = f'{state}{label_count}'
                label_count += 1
                consecutive_count = 0

        else:
            consecutive_count = 0

    return df

# TODO: move brain_state_filter to eeg_class.EegData — it operates only on EEG/velocity DataFrames
def brain_state_filter(velo_eeg_df: pd.DataFrame, states:list) -> tuple:
    """
    sets up boolean masks for brain states

    Paramaters:
    ===========
    velo_eeg_df: pandas DataFrame
        contains a 'filtered_velo' columns with the velocity and columns
        with brain states. See eeg.import_eeg()
    states: list
        list of the possible brain states: NREM, REM, awake
        
    Return:
    =======
    filters: dictionary
        the keys correspond to boolean values
    states: list
        the states used as an input plus 'locomotion'

    """
    l1 = ['NREM', 'REM', 'awake']
    filters = {}
    if Counter(states) == Counter(l1):
        logging.info("Making filters for %s and locomotion", l1)
        awake = ((velo_eeg_df['NREM']==False) &
             (velo_eeg_df['REM']==False) &
             (velo_eeg_df['filtered velo']<IMMOBILITY_VELOCITY_THRESHOLD))
        nrem = ((velo_eeg_df['NREM']) &
                (velo_eeg_df['filtered velo']<IMMOBILITY_VELOCITY_THRESHOLD))
        rem = ((velo_eeg_df['REM']) &
               (velo_eeg_df['filtered velo']<IMMOBILITY_VELOCITY_THRESHOLD))
        locomotion = ((velo_eeg_df['NREM']==False) &
              (velo_eeg_df['REM']==False) &
              (velo_eeg_df['filtered velo']>=IMMOBILITY_VELOCITY_THRESHOLD))
        filters = {'awake':awake,
                  'NREM': nrem,
                  'REM':rem,
                  'locomotion':locomotion}        
    else:
        logging.info("no REM, making filters for NREM, awake and locomotion")
        awake = ((velo_eeg_df['NREM']==False) &
             (velo_eeg_df['REM']==False) &
             (velo_eeg_df['filtered velo']<IMMOBILITY_VELOCITY_THRESHOLD))
        nrem = ((velo_eeg_df['NREM']) &
                (velo_eeg_df['filtered velo']<IMMOBILITY_VELOCITY_THRESHOLD))
        locomotion = ((velo_eeg_df['NREM']==False) &
              (velo_eeg_df['REM']==False) &
              (velo_eeg_df['filtered velo']>=IMMOBILITY_VELOCITY_THRESHOLD))
        filters = {'awake': awake,
                  'nrem': nrem,
                  'locomotion':locomotion}
    states.append('locomotion')
    return filters, states

def upper(df: pd.DataFrame) -> pd.DataFrame:
    '''Returns the upper triangle of a correlation matrix.
    You can use scipy.spatial.distance.squareform to recreate
    matrix from upper triangle.
    
    Source: towardsdatascience.com
        how-to-measure-similarity-between-two-correlation-matrices

    Args:
      df: pandas or numpy correlation matrix

    Returns:
        pd.DataFrame with the values from the upper triangle
    '''
    if isinstance(df, pd.DataFrame):
        df = df.values
    elif not isinstance(df, np.ndarray):
        raise TypeError('Must be pandas or numpy correlation matrix')
    mask = np.triu_indices(df.shape[0], k=1)
    return df[mask]

def data_loader(data_dir: str, data_type: str, mouse_id: str,
                 day: str, session_id: str) -> dict:
    """loads calcium, spike, eeg, cellular data from one session
    Parameters:
    ===========
    data_dir: str
    path like object pointing to the location of the data
    mouseID, day, sessionID: str
    data_type: str
    'dfof' or 'spikes'

    Retrurns:
    =========
    dictionary containing the data
    """
    if data_type != ('dfof' or 'spikes'):
        raise ValueError('data_type must be "dfof" or "spikes"')
    data_loc = Path(data_dir).joinpath(mouse_id, day, session_id)
    data = pd.read_csv(Path(data_loc).joinpath(data_type+'.csv')).set_index('roi_label')

    # loading stat results for significantly up and downregulated cells
    # during NREM
    sig_cells = pd.read_csv(Path(data_loc).joinpath('Significant_paired_DABEST_NREM.csv'))
    logging.info("number of cells in this recording: %d", len(sig_cells))

    # loadig EEG and behavior data
    eeg_velocity = pd.read_csv(Path(data_loc).joinpath('velo_eeg.csv'))
    return { data_type: data,
            'significant_cells': sig_cells,
            'eeg_velocity':eeg_velocity}

def significant_cells(cell_data: pd.DataFrame, state: str) -> dict:
    """Returns the list of significant cells for a given state
    
    Parameters:
    ===========
    cell_data: pd.DataFrame
        this is coming from the data_loader function    
    state: str
        brain state, e.g. 'NREM', 'REM', 'awake'

    Return:
    =======
    ind_list: dict
    	dict of significant roi_label indices
    """
    cell_data.drop(labels='Unnamed: 0', inplace=True, axis=1)
    cell_data.set_index('roi_label', inplace=True)
    up_ind_list = list(cell_data[
        (cell_data['Direction']=='Upregulated') &
                             (cell_data['State']=='dfof '+state)].index)
    down_ind_list = list(cell_data[
        (cell_data['Direction']=='Downregulated') &
                             (cell_data['State']=='dfof '+state)].index)
    return {'Upregulated': up_ind_list,
            'Downregulated': down_ind_list}
