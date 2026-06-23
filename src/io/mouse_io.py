"""Data class dealing with mouse data.
author: @gergelyturi
date: 2023-10-16
204-11-09: major refactoring. this used to be the ImagingData class,
now it is the MouseData class"""

import os
from dataclasses import dataclass
from os import walk
from pathlib import Path
from typing import List, Union

try:
    from src.config import SLEEP_DATA_ROOT as _DEFAULT_ROOT
except ImportError:
    from config import SLEEP_DATA_ROOT as _DEFAULT_ROOT


@dataclass
class MouseData:
    """
    Class representing data folders belonging to a mouse.
    
    Attributes
    ----------
    mouse_id : Union[str, Path]
        The identifier for the mouse.
    root_folder : Union[str, Path]
        The root directory where the mouse data is stored.
    """

    mouse_id: Union[str, Path]
    root_folder: Union[str, Path] = _DEFAULT_ROOT

    def __post_init__(self):
        """
        Initializes the mouse data object.
        
        Raises
        ------
        ValueError
            If `mouse_id` or `root_folder` is not provided.
        """
        if not self.mouse_id:
            raise ValueError("Mouse ID must be a non-empty string")
        if not self.root_folder:
            raise ValueError("root_folder must be specified")

        # Ensure mouse_id and root_folder are always Path objects
        if isinstance(self.mouse_id, str):
            self.mouse_id = Path(self.mouse_id)
        if isinstance(self.root_folder, str):
            self.root_folder = Path(self.root_folder)

        # Create the path to the mouse folder
        self.mouse_folders = self.root_folder / self.mouse_id

    def _find_folders(self, condition: callable) -> List[Path]:
        """
        General helper function to find folders that meet a certain condition.

        Parameters
        ----------
        condition : callable
            A function that takes a folder name and returns a boolean indicating whether
            the folder meets the desired condition.

        Returns
        -------
        List[Path]
            A list of paths to the folders that meet the condition.
        
        Raises
        ------
        ValueError
            If no folders meeting the condition are found.
        """
        if not self.mouse_folders.exists():
            raise FileNotFoundError(
                f"Expected mouse data directory at {self.mouse_folders} — "
                f"check mouse_id or set the SLEEP_DATA_ROOT environment variable."
            )
        folders = []
        for dirpath, dirnames, _ in walk(self.mouse_folders):
            for dirname in dirnames:
                folder_path = Path(dirpath) / dirname
                if condition(dirname):
                    folders.append(folder_path)
        if not folders:
            raise ValueError(f"No folders meeting the condition were found in {self.mouse_folders}")
        return folders

    def find_s2p_folders(self) -> List[Path]:
        """
        Finds all Suite2p folders for a given mouse.

        Returns
        -------
        List[Path]
            A list of all Suite2p folders found in the mouse folder.
        
        Raises
        ------
        ValueError
            If no Suite2p folders are found.
        """
        return self._find_folders(lambda dirname: dirname.startswith("suite2p"))

    def find_tseries_folders(self) -> List[Path]:
        """
        Finds all TSeries folders for a given mouse.

        Returns
        -------
        List[Path]
            A list of all TSeries folders found in the mouse folder.
        
        Raises
        ------
        ValueError
            If no TSeries folders are found.
        """
        return self._find_folders(lambda dirname: dirname.startswith("TSeries") and not dirname.endswith(".sima"))

    def find_sima_folders(self) -> List[Path]:
        """
        Finds all .sima folders for a given mouse.

        Returns
        -------
        List[Path]
            A list of all .sima folders found in the mouse folder.
        
        Raises
        ------
        ValueError
            If no .sima folders are found.
        """
        return self._find_folders(lambda dirname: dirname.endswith(".sima"))

    def find_eeg_folders(self) -> List[Path]:
        """
        Finds all EEG folders for a given mouse.

        Returns
        -------
        List[Path]
            A list of all EEG folders found in the mouse folder.
        
        Raises
        ------
        ValueError
            If no EEG folders are found.
        """
        return self._find_folders(lambda dirname: dirname == "eeg")

    def find_behavior_folders(self) -> List[Path]:
        """
        Finds all behavior folders for a given mouse.

        Returns
        -------
        List[Path]
            A list of all behavior folders found in the mouse folder.
        
        Raises
        ------
        ValueError
            If no behavior folders are found.
        """
        return self._find_folders(lambda dirname: dirname == "behavior")
