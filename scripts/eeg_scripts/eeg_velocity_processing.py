"""This script takes scored data from a csv file and adds velocity data to it.
Finally it calculates different brain states
Author: Gergely Turi
gt2253@cumc.columbia.edu
v0.0 10/11/2023
11/27/2023: This is done: the script generates an extra line at the end if it uses interpolation. need to track down why
but need to test more
3/13/2024: moving away from csv and adding the functionality to use json files.
3/16/2024: NOTE: something is wrong. it seems that the lenght of ` im_exp.signals().shape[1]`
    is sometimes not the same as the number of acquired images. see 126031_1/11_23.
5/4/2024: changed the interpolation method to nearest neighbor interpolation. The problem described above is
still not solved.
11/10/2024: the script was moved from lab3 to sleep repo in a hope that the interpolation problem will be solved
    also, implemented a resampling instead of interpolation as it better suits categorical data.
"""

import json
import logging
from argparse import ArgumentParser
from os.path import dirname, join
from pathlib import Path

import numpy as np
import pandas as pd

from src.io import eeg_io as eeg
from src.io import imaging_io as ic


def resample_categorical(data: pd.Series, new_length: int) -> pd.Series:
    """
    Resample categorical data by selecting the nearest original values to match the new length.

    Parameters:
    ----------
    data : pd.Series
        The original categorical data to be resampled.
    new_length : int
        The desired length of the resampled data.

    Returns:
    -------
    pd.Series
        A resampled version of the categorical data with the specified length.
    """
    original_indices = np.linspace(0, len(data) - 1, new_length, dtype=int)
    resampled_data = data.iloc[original_indices]
    return resampled_data.reset_index(drop=True)

def main(eeg_folder_path, file_name: str = "sleep.csv"):
    try:
        # Instantiate EegData object
        eeg_data = eeg.EegData(eeg_folder_path)
        eeg_df = eeg_data.import_scored_eeg(file_name, processed=True)
    except FileNotFoundError:
        logging.error("No scored data found or the file is not named %s.", file_name)
        return

    # Attempt to load velocity data from either CSV or JSON
    activity_file_names = ["filtered_velo.csv", "filtered_velocity.json"]
    velocity_loaded = False
    for activity_file_name in activity_file_names:
        try:
            activity_file_path = join(dirname(dirname(eeg_folder_path)), "behavior", activity_file_name)
            logging.info("Attempting to load velocity data from %s...", activity_file_path)
            if activity_file_name.endswith(".csv"):
                velocity = pd.read_csv(activity_file_path)
            elif activity_file_name.endswith(".json"):
                with open(activity_file_path, "r") as f:
                    velocity_json = json.load(f)
                    velocity = pd.DataFrame(velocity_json, columns=["velocity"])
            velocity_loaded = True
            break  # File loaded successfully, exit the loop
        except FileNotFoundError:
            continue  # Try the next file name
    
    if not velocity_loaded:
        logging.error("No velocity data found in CSV or JSON format.")
        return

    # Resample if lengths are not equal
    if len(eeg_df) != len(velocity):
        logging.warning(
            "The length of the two files are not equal. "
            "Attempting to resample EEG data to match the length of the imaging/velocity data."
        )
        raw_eeg = eeg_data.import_scored_eeg(file_name, processed=False)

        # Handling single plane vs. multiplane data
        tseries_folder = Path(eeg_folder_path).parents[1] 
        imaging_exp = ic.Imaging(tseries_folder)
        metadata = imaging_exp.get_imaging_metadata()
        
        if metadata.get("sequence_type") == "single plane":
            new_length = int(metadata.get("number of images", 0))
        elif metadata.get("sequence_type") == "multi plane":
            new_length = int(metadata.get("number of sequences", 0))
        else:
            logging.error("Unknown imaging sequence type.")
            return

        # Resample the EEG data to match the new length
        resampled_scores = resample_categorical(raw_eeg["score"], new_length)
        resampled_eeg = pd.DataFrame(resampled_scores, columns=["score"])
        resampled_eeg["score"] = resampled_eeg["score"].astype(int)

        # Add the other columns based on the resampled scores
        resampled_eeg["awake"] = (resampled_eeg["score"] == 0).astype(int)
        resampled_eeg["NREM"] = (resampled_eeg["score"] == 1).astype(int)
        resampled_eeg["REM"] = (resampled_eeg["score"] == 2).astype(int)
        resampled_eeg["other"] = (resampled_eeg["score"] == 3).astype(int)

        eeg_df = resampled_eeg

    # Concatenate velocity and EEG data
    velo_eeg_df = pd.concat([velocity.reset_index(drop=True), eeg_df], axis=1)

    # Save the combined DataFrame to a CSV file
    output_file_path = join(eeg_folder_path, "velo_eeg.csv")
    velo_eeg_df.to_csv(output_file_path, index=False)

    logging.info("Brain state filter is done.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = ArgumentParser()
    parser.add_argument(
        "eeg_folder",  
        type=str,
        help="Path to the EEG folder to process.",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="sleep.csv",
        help="Name of the file to process.",
    )
    args = parser.parse_args()

    # Call main function with arguments
    main(args.eeg_folder, args.file_name)

