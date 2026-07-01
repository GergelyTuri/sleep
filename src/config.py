"""Centralized analysis parameters for the sleep project.

Scientific constants (reproducibility-critical) are defined directly in this file
and kept under version control.  Machine-local paths live in config_local.py
(gitignored); copy config_local.py.example to create it.  All names are
re-exported from this module so existing ``from config import X`` /
``from src.config import X`` call sites require no changes.
"""

import os
import pathlib as _pathlib

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

# --------------- Machine-local paths (config_local.py — gitignored) ---------------
# Copy src/config_local.py.example to src/config_local.py and fill in your values.
# config_local.py is gitignored — never commit it.
try:
    from .config_local import *  # noqa: F401,F403 — package mode (editable install)
except (ImportError, SystemError):
    try:
        from config_local import *  # noqa: F401,F403 — standalone / Colab mode
    except ImportError:
        pass  # no config_local.py present — path variables fall back to env vars or None

# Env-var overrides win over config_local (allows SLURM/HPC inline override).
_env_root = os.environ.get("SLEEP_DATA_ROOT")
if _env_root:
    SLEEP_DATA_ROOT = _env_root
SLEEP_DATA_ROOT = (
    _pathlib.Path(SLEEP_DATA_ROOT) if globals().get("SLEEP_DATA_ROOT") else None
)

_env_gdrive_data = os.environ.get("SLEEP_GDRIVE_DATA_PATH")
if _env_gdrive_data:
    GDRIVE_DATA_PATH = _env_gdrive_data
elif "GDRIVE_DATA_PATH" not in globals():
    GDRIVE_DATA_PATH = None

_env_gdrive_root = os.environ.get("SLEEP_GDRIVE_ROOT")
if _env_gdrive_root:
    GDRIVE_ROOT = _env_gdrive_root
elif "GDRIVE_ROOT" not in globals():
    GDRIVE_ROOT = None


def get_data_root() -> _pathlib.Path:
    """Return SLEEP_DATA_ROOT as a validated Path, or raise RuntimeError.

    Call this at the start of any script or notebook that needs real data.
    MouseData.__post_init__ delegates here when no explicit root_folder is given.
    """
    if not SLEEP_DATA_ROOT:
        raise RuntimeError(
            "SLEEP_DATA_ROOT is not configured. "
            "Copy src/config_local.py.example to src/config_local.py and set the path, "
            "or export SLEEP_DATA_ROOT=/your/data/path."
        )
    p = _pathlib.Path(SLEEP_DATA_ROOT)
    if not p.exists():
        raise RuntimeError(
            f"SLEEP_DATA_ROOT does not exist: {p}. "
            "Set it in config_local.py (copy from config_local.py.example) "
            "or export SLEEP_DATA_ROOT=/your/data/path."
        )
    return p
