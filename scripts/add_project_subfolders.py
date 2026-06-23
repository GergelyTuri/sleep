import argparse

from src.io.mouse_io import MouseData
from src.io.sleep_experiment import SleepExperiment


def main():
    """
    This script takes a mouse ID as input and creates the necessary folder structure for the experiment.

    Usage: python add_project_subfolders.py <mouse_id>

    Args:
        mouse_id (str): The ID of the mouse for which the sleep experiment data will be processed.
    """
    parser = argparse.ArgumentParser(description="Process the sleep experiment data.")
    parser.add_argument("mouse_id", type=str, help="Enter the mouse ID")
    args = parser.parse_args()

    imaging_data = MouseData(args.mouse_id)
    tseries_folders = imaging_data.find_tseries_folders()

    for tseries_folder in tseries_folders:
        experiment = SleepExperiment(args.mouse_id, tseries_folder=tseries_folder)
        experiment.create_folder_structure()


if __name__ == "__main__":
    main()
