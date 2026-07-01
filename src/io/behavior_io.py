import json
import logging
from dataclasses import dataclass
from os.path import join
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from . import imaging_io as ic

try:
    from src.config import IMMOBILITY_VELOCITY_THRESHOLD
except ImportError:
    from config import IMMOBILITY_VELOCITY_THRESHOLD

logger = logging.getLogger(__name__)


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
        threshold: float = IMMOBILITY_VELOCITY_THRESHOLD,
        min_duration: float = 1.0,
        min_periods: int = 1,
        center: bool = True,
    ) -> pd.Series:
        """
        Define time periods of mobility based on a rolling window of speed.

        A frame is classified mobile if the rolling-maximum speed over ``min_duration``
        seconds meets or exceeds ``threshold`` (consistent with
        ``brain_state_filter``'s ``>= IMMOBILITY_VELOCITY_THRESHOLD`` boundary).

        Default values for min_duration are taken from:
        Stefanini et al., 2018 (https://doi.org/10.1101/292953)

        Args:
            velocity (np.ndarray): Per-frame velocity in **cm/s** (signed; positive =
                forward, negative = backward), as produced by
                ``BehaviorData.resample_to_imaging()``.  ``abs(velocity)`` is applied
                internally so backward motion is classified mobile, not immobile.
            threshold (float): Speed threshold in **cm/s** at or above which the mouse
                is considered mobile.  Defaults to
                ``config.IMMOBILITY_VELOCITY_THRESHOLD`` (1.0 cm/s), which sits above
                the encoder-jitter floor (~0.5 cm/s) and below real running (~10 cm/s).
            min_duration (float): Minimum duration (seconds) of the rolling window.
            min_periods (int): Minimum observations required in the rolling window.
            center (bool): Whether to center the rolling window.

        Returns:
            pd.Series: Boolean series — True = mobile, False = immobile.
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

        # Rectify to speed so backward motion (negative velocity) is classified mobile.
        speed = np.abs(np.asarray(velocity, dtype=float))
        speed_series = pd.Series(speed)
        window_size = int(framerate * min_duration)

        if window_size < 1:
            raise ValueError(f"Calculated window size must be at least 1, but got {window_size}.")

        rolling_max_speed = speed_series.rolling(
            window_size, min_periods=min_periods, center=center
        ).max()

        # >= matches brain_state_filter's locomotion boundary (>= IMMOBILITY_VELOCITY_THRESHOLD)
        mobility = (rolling_max_speed >= threshold).astype(bool)

        return mobility

    def resample_to_imaging(
        self,
        velocity_ts: np.ndarray,
        imaging_fps: float,
        n_frames: int,
    ) -> np.ndarray:
        """Resample an event-driven velocity time series onto the imaging frame grid.

        Replaces the lab3 ``format_behavior_data(sampling_interval=frame_period)``
        call.  The input must carry per-event timestamps because BehaviorMate is
        event-driven (no fixed sampling rate); endpoints alone are not sufficient
        to reconstruct the time base.

        Parameters
        ----------
        velocity_ts : np.ndarray, shape (N, 2)
            Column 0: event timestamps in seconds from recording start.
            Column 1: velocity value at each event.
            Timestamps must be strictly monotonically increasing.
        imaging_fps : float
            Imaging frame rate in Hz, supplied by the caller.
        n_frames : int
            Number of imaging frames (output length).

        Returns
        -------
        np.ndarray, shape (n_frames,)
            Velocity resampled onto the imaging frame grid.  Unfiltered — apply
            any smoothing (e.g. Gaussian) after calling this method.

        Raises
        ------
        ValueError
            If ``velocity_ts`` is not shape ``(N, 2)``, timestamps are not
            strictly increasing, ``imaging_fps`` or ``n_frames`` are not
            positive, or the behavior recording ends before the last imaging
            frame (beyond a one-frame tolerance).
        """
        if velocity_ts.ndim != 2 or velocity_ts.shape[1] != 2:
            raise ValueError(
                "velocity_ts must have shape (N, 2); got %s" % (velocity_ts.shape,)
            )
        if imaging_fps <= 0:
            raise ValueError(
                "imaging_fps must be positive; got %s" % imaging_fps
            )
        if n_frames <= 0:
            raise ValueError(
                "n_frames must be positive; got %s" % n_frames
            )

        behavior_times = velocity_ts[:, 0]
        velocity_values = velocity_ts[:, 1]

        if np.any(np.diff(behavior_times) <= 0):
            raise ValueError(
                "velocity_ts timestamps are not strictly increasing"
            )

        target_times = np.arange(n_frames) / imaging_fps
        tolerance = 1.0 / imaging_fps

        if target_times[-1] > behavior_times[-1] + tolerance:
            raise ValueError(
                "Behavior ends %.2f s before last imaging frame "
                "(behavior ends at %.2f s, last frame at %.2f s). "
                "Padding for short behavior is not yet implemented."
                % (
                    target_times[-1] - behavior_times[-1],
                    behavior_times[-1],
                    target_times[-1],
                )
            )

        return np.interp(target_times, behavior_times, velocity_values)
