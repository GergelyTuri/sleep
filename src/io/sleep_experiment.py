"""
Class for sleep experiment data
"""

import logging
from dataclasses import dataclass
from os import makedirs
from os.path import join, exists, isdir
from os import listdir

logging.basicConfig(level=logging.INFO)


@dataclass
class SleepExperiment:
    """Class for sleep experiment data"""

    mouse_id: str
    tseries_folder: str = None

    def __post_init__(self):
        """
        Initializes the SleepExperiment object.

        """
        if not self.mouse_id:
            raise ValueError("Mouse ID must be a non-empty string")

    def create_folder_structure(self, subdirectories=None) -> None:
        """
        Creates a custom folder structure for the experiment within the .sima folder if it exists.
        """
        sima_folder_path = self._construct_sima_folder_path()
        if sima_folder_path is None:
            logging.warning(f"No .sima folder found in {self.tseries_folder}.")
            return

        if subdirectories is None:
            subdirectories = ["behavior", "eeg", "plots"]

        for dir_name in subdirectories:
            self._create_subdirectory(sima_folder_path, dir_name)

    def _create_subdirectory(self, sima_folder_path: str, dir_name: str) -> None:
        """
        Create a subdirectory in the specified SIMA folder path.

        Args:
            sima_folder_path (str): The path to the SIMA folder.
            dir_name (str): The name of the subdirectory to create.

        Returns:
            None
        """
        # Extract the TSeries folder name from the full path
        path = join(sima_folder_path, dir_name)

        try:
            if not exists(path):
                makedirs(path)
                logging.info(f"Created directory: {path}")
            else:
                logging.info(f"Directory already exists: {path}")
        except OSError as error:
            logging.error(f"Error creating directory {path}: {error}")

    def _construct_sima_folder_path(self) -> str:
        """
        Constructs the path to the .sima folder, accounting for variations in naming.

        Returns:
            str: The path to the .sima folder, or None if not found.
        """
        tseries_folder_name = self.tseries_folder.rstrip("/").split("/")[-1]

        # List all subdirectories in the TSeries folder
        try:
            subdirs = [
                d
                for d in listdir(self.tseries_folder)
                if isdir(join(self.tseries_folder, d))
            ]
        except FileNotFoundError:
            return None

        # Find a subdirectory that matches the .sima pattern
        for subdir in subdirs:
            if subdir.startswith(tseries_folder_name) and subdir.endswith(".sima"):
                return join(self.tseries_folder, subdir)

        return None
