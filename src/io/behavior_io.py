import json
from dataclasses import dataclass
from os.path import join
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from . import imaging_io as ic


@dataclass
class BehaviorData:
    """Class for behavior data.
    Initialized with the path to the behavior folder.
    """

    behavior_dir: Union[str, Path]

    def __post_init__(self):
        if isinstance(self.behavior_dir, str):
            self.behavior_dir = Path(self.behavior_dir)


    def load_processed_velocity(self, file_name: str = "filtered_velocity.json") -> np.ndarray:
        """
        Return the processed velocity of the mouse.

        Parameters:
        - file_name (str): The name of the velocity file.

        Returns:
        - np.ndarray: The processed velocity as a NumPy array.

        Raises:
        - FileNotFoundError: If the velocity file is not found in the specified folder.
        - ValueError: If the velocity file cannot be read as valid JSON.
        """
        try:
            with open(join(self.behavior_dir, file_name), "r") as f:
                processed_velocity = np.array(json.load(f))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find processed velocity file '{file_name}' in {self.behavior_dir}."
            )
        except json.JSONDecodeError:
            raise ValueError(
                f"Could not decode JSON from velocity file '{file_name}' in {self.behavior_dir}."
            )
        return processed_velocity

    def define_mobility(
        self,
        velocity: np.ndarray,        
        threshold: float = 1.0,
        min_duration: float = 1.0,
        min_periods: int = 1,
        center: bool = True,
    ) -> pd.Series:
        """
        Define time periods of immobility based on a rolling window of velocity.

        A Mouse is considered immobile if velocity has not exceeded the threshold for the
        previous min_duration seconds.

        Default values for min_duration and threshold are taken from:
        Stefanini et al., 2018 (https://doi.org/10.1101/292953)

        Args:
            velocity (np.ndarray): The filtered and processed velocity of the mouse.
            threshold (float): The threshold value for defining immobility.
            min_duration (float): The minimum duration (in seconds) to consider the mouse immobile.
            min_periods (int): The minimum number of observations required to be considered immobile.
            center (bool): Whether to center the rolling window.

        Returns:
            pd.Series: A one-dimensional series of booleans, where False signifies immobile
            times and True signifies mobile times.
        """
        # Getting the framerate from the imaging metadata.
        # Assumes behavior_dir is <TSeries>/.sima/behavior/ (parents[1] = TSeries root).
        tSeries_path = (self.behavior_dir).parents[1]
        if not tSeries_path.exists():
            raise FileNotFoundError(
                f"Expected TSeries folder at {tSeries_path} — "
                f"verify behavior_dir sits inside <TSeries>/.sima/behavior/."
            )
        imaging = ic.Imaging(tSeries_path)
        imaging_metadata = imaging.get_imaging_metadata()

        sequence_type = imaging_metadata.get("sequence_type")
        if sequence_type == "single plane":
            framerate = float(round(imaging_metadata.get("fps", 0), 2))
        elif sequence_type == "multi plane":
            framerate = imaging.multiplane_frame_rate()
            if isinstance(framerate, str) or framerate == 0.0:
                raise ValueError(f"Invalid frame rate obtained for multiplane imaging: {framerate}")
        else:
            raise ValueError(f"Unknown imaging sequence type: {sequence_type}")

        if not (1.0 <= framerate <= 100.0):
            raise ValueError(
                f"Frame rate {framerate} Hz is outside the plausible range (1–100 Hz). "
                f"Check 'fps' in imaging_metadata.json at {tSeries_path}."
            )

        # Calculating mobile/immobile periods
        velocity_series = pd.Series(velocity).astype(float)
        window_size = int(framerate * min_duration)
        
        # Ensure window_size is at least 1 to avoid rolling errors
        if window_size < 1:
            raise ValueError(f"Calculated window size must be at least 1, but got {window_size}.")

        rolling_max_vel = velocity_series.rolling(
            window_size, min_periods=min_periods, center=center
        ).max()

        # Define mobility/immobility periods
        mobility = (rolling_max_vel > threshold).astype(bool)

        return mobility
