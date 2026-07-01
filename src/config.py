"""Centralized analysis parameters for the sleep project.

All numeric defaults used as function/class constants are gathered here so
a researcher can inspect and override them in one place without hunting
across modules.
"""

import os

# --------------- Signal filtering (classes/filters.py) ---------------
MAXMIN_WINDOW = 300       # default rolling window size in frames
MAXMIN_SIGMA = 5          # default Gaussian smoothing sigma in frames
QUANTILE_BASELINE = 0.08  # baseline quantile for smooth_quantile_filter
FILTER_MIN_PERIODS = 0.2  # fraction of window that must be non-NaN

# --------------- dF/F (classes/dfof.py) ---------------
NPIL_COEFF = 0.7             # neuropil subtraction coefficient (Suite2p convention)
SUITE2P_DFOF_WINDOW = 600    # baseline window in frames (~60 s at ~10 fps)
SUITE2P_DFOF_SIGMA = 10      # Gaussian smoothing sigma in frames
SUITE2P_DFOF_MIN_PERIODS = 0.2
JIA_T1 = 90    # initial smoothing window (~3 s at 30 fps)
JIA_T2 = 1800  # rolling-minimum baseline window (~60 s at 30 fps)

# --------------- Epoch thresholds (calc_module.py, clustering.py) ---------------
MIN_EPOCH_FRAMES = 100               # minimum consecutive frames to label a state epoch
MIN_SLEEP_EPOCH_FRAMES = 600         # minimum frames for a valid sleep epoch
IMMOBILITY_VELOCITY_THRESHOLD = 1.0  # speed (cm/s) at or above which mouse is mobile

# --------------- Signal analysis (frequency_psd.py) ---------------
MAX_AUTOCORR_LAGS = 500  # number of autocorrelation lags to retain

# --------------- EEG scoring (classes/eeg_class.py) ---------------
KNOWN_EEG_CODES = frozenset({0, 1, 2, 3})  # 0=awake, 1=NREM, 2=REM, 3=other

# --------------- Data paths (override via environment variables) ---------------
SLEEP_DATA_ROOT = os.environ.get("SLEEP_DATA_ROOT", "/mnt/d/from_data/invivo_DATA/sleep")
GDRIVE_DATA_PATH = os.environ.get(
    "SLEEP_GDRIVE_DATA_PATH",
    "/gdrive/Shareddrives/Turi_lab/Data/Sleep/2p/Analysis/data",
)
GDRIVE_ROOT = os.environ.get("SLEEP_GDRIVE_ROOT", "/gdrive/Shareddrives/Turi_lab/Data/")
