"""Convenience loader that assembles all data for one recording session.

Typical usage
-------------
>>> from src.io.session_io import load_session
>>> session = load_session(
...     sima_folder="/data/mouse01/TSeries-20231012.sima",
...     s2p_folder="/data/mouse01/suite2p",
... )
>>> session.dfof      # pd.DataFrame (n_rois × n_frames)
>>> session.eeg       # pd.DataFrame with awake/NREM/REM columns
>>> session.mobility  # pd.Series of booleans (True = mobile)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SessionData:
    """All data for a single recording session."""

    sima_folder: Path
    mouse_id: str = ""
    dfof: Optional[pd.DataFrame] = None
    eeg: Optional[pd.DataFrame] = None
    mobility: Optional[pd.Series] = None
    velocity: Optional[np.ndarray] = None

    def __post_init__(self):
        self.sima_folder = Path(self.sima_folder)


def load_session(
    sima_folder,
    mouse_id: str = "",
    s2p_folder=None,
    dfof_strategy=None,
    eeg_file: str = "sleep.csv",
    velocity_file: str = "filtered_velocity.json",
) -> SessionData:
    """Load all data for one session from the standard folder layout.

    Parameters
    ----------
    sima_folder : str or Path
        Path to the TSeries ``.sima`` directory.
    mouse_id : str, optional
        Mouse identifier string.
    s2p_folder : str or Path, optional
        Path to the Suite2p output directory. When provided, Suite2p signals
        are loaded and dF/F is computed.
    dfof_strategy : DFOFStrategy, optional
        dF/F calculation strategy instance.  Defaults to ``JiaDFOF()``.
    eeg_file : str
        Filename for the scored EEG CSV, relative to ``sima_folder/eeg/``.
    velocity_file : str
        Filename for the filtered velocity JSON, relative to
        ``sima_folder/behavior/``.

    Returns
    -------
    SessionData
    """
    sima_folder = Path(sima_folder)
    session = SessionData(sima_folder=sima_folder, mouse_id=mouse_id)

    # --- EEG -----------------------------------------------------------------
    eeg_dir = sima_folder / "eeg"
    if eeg_dir.exists():
        from src.io.eeg_io import EegData
        try:
            session.eeg = EegData(eeg_dir).import_scored_eeg(eeg_file)
        except FileNotFoundError:
            logging.warning("EEG file not found in %s", eeg_dir)

    # --- Behavior ------------------------------------------------------------
    behavior_dir = sima_folder / "behavior"
    if behavior_dir.exists():
        from src.io.behavior_io import BehaviorData
        try:
            bd = BehaviorData(behavior_dir)
            session.velocity = bd.load_processed_velocity(velocity_file)
            session.mobility = bd.define_mobility(session.velocity)
        except (FileNotFoundError, ValueError) as exc:
            logging.warning("Could not load behavior data: %s", exc)

    # --- Suite2p + dF/F ------------------------------------------------------
    if s2p_folder is not None:
        from src.io.suite2p_io import Suite2p
        from src.preprocessing.dfof import JiaDFOF
        try:
            s2p = Suite2p(s2p_folder)
            signal = pd.DataFrame(s2p.get_cells())
            strategy = dfof_strategy if dfof_strategy is not None else JiaDFOF()
            session.dfof = strategy.calculate(signal=signal)
        except Exception as exc:
            logging.warning("Could not compute dF/F: %s", exc)

    return session
