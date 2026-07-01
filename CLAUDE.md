# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Analysis codebase for a neuroscience sleep project. It processes in-vivo two-photon calcium imaging data (via Suite2p), EEG/EMG sleep-state scoring, and behavioral velocity data from mice. Notebooks are the primary analysis interface; `src/` holds the reusable library code they import.

## Environment Setup

```bash
conda env create -f environment.yaml
conda activate sleep
pip install -e .
```

The package is installed as `sleep` (editable via `pyproject.toml`). Pip-only extras are in `requirements.txt`; install both.

### Machine-local config (required on every new machine)

`src/config.py` holds scientific constants under version control. Machine-local
paths live in `src/config_local.py`, which is **gitignored and never committed**.
Copy the example and set your data root before running any analysis:

```bash
cp src/config_local.py.example src/config_local.py
# edit src/config_local.py — set SLEEP_DATA_ROOT to your data directory
```

On **UTSW BioHPC**: code lives in `$HOME`; data lives on a project/scratch
allocation. Point `SLEEP_DATA_ROOT` there, not at `$HOME`.

Alternatively, set the environment variable inline (e.g. in a SLURM job script):

```bash
export SLEEP_DATA_ROOT=/project/tlab/invivo_DATA/sleep
```

The env var always wins over `config_local.py`.  If neither is set,
`MouseData()` raises a clear `RuntimeError` on construction.

## Running Notebooks

```bash
jupyter notebook
```

Notebook-based tests live in `tests/` and are run manually in Jupyter. A shell-runnable pytest suite also exists:

```bash
pytest tests/test_unit.py -v
```

## Project Structure

```
src/
  config.py              # Scientific constants (versioned). Re-exports machine-local paths from config_local.py.
  config_local.py        # Machine-local paths — GITIGNORED, never committed. Copy from config_local.py.example.
  config_local.py.example  # Committed template for config_local.py.
  logging_setup.py       # Empty stub — handlers configured per-script via logging.basicConfig()
  io/                    # Data loading — one file per data source
    suite2p_io.py        # Suite2p class (loads F.npy, Fneu.npy, spks.npy, iscell.npy)
    imaging_io.py        # Imaging class (reads imaging_metadata.json)
    behavior_io.py       # BehaviorData class (velocity JSON, mobility epochs)
    eeg_io.py            # EegData class (scored EEG CSVs)
    mouse_io.py          # MouseData class (filesystem walk for a mouse)
    sleep_experiment.py  # SleepExperiment class (creates .sima subfolder layout)
    session_io.py        # load_session() → SessionData (combines all sources in one call)
  preprocessing/
    filters.py           # maxmin_filter baseline
    dfof.py              # Suite2pDFOF, JiaDFOF (strategy-pattern dF/F calculators)
    event_detection.py   # EventDetection, Denoiser (calcium transient detection)
  analysis/
    calc_module.py       # Epoch labeling, brain_state_filter helpers
    clustering.py        # Hierarchical clustering, sleep-epoch utilities
    frequency_psd.py     # freq_calc, calculate_psd, calculate_autocorrelations
    custom_statistics.py # Mann-Whitney U, Bonferroni correction
    estimation_stats.py  # Cohen's d, DABEST estimation stats
    xcorr/               # Cross-correlation analysis
  visualization/
    full_frame_analysis.py   # Butterworth filters, autocorrelation plotting
    frequency_psd_plots.py   # plot_psd, spectral_density_plot
  db/
    database.py          # ExperimentDatabase — MySQL wrapper via pymysql
  colab/                 # Colab-only utilities (import google.colab; guarded at top)
    google_utils.py
    google_drive.py
notebooks/               # Analysis organized by topic
  behavior/
    process_velocity.ipynb   # Derives signed velocity in cm/s from treadmillDy; saves [[time_s, velocity_cms], ...]
scripts/
  add_project_subfolders.py        # Creates behavior/eeg/plots subfolders for a mouse's .sima dirs
  add_time_avg.py                  # Adds time-averaged image to suite2p data
  behavior_scripts/
    export_mobility_immobility.py  # Exports mobility/immobility epochs → JSON
    process_tdml_behavior_data.py  # Parses .tdml/.vr files → .json (default) or .pkl
  eeg_scripts/
    eeg_velocity_processing.py     # Merges scored EEG with velocity data → velo_eeg.csv
    process_matlab_files.py        # Extracts sleep state scores from .mat files → CSV
tests/
  behavior/              # Sample .tdml files used by pytest integration tests
  test_unit.py           # pytest suite (includes TestProcessJsonBehaviorData)
```

### Key classes and their import paths

- **`MouseData`** (`src.io.mouse_io`) — entry point for a mouse's data. Takes `mouse_id` and optional `root_folder`. If `root_folder` is omitted it is resolved via `config.get_data_root()` (reads `SLEEP_DATA_ROOT` from `config_local.py` or the `SLEEP_DATA_ROOT` env var); raises `RuntimeError` immediately on construction if neither is set or the path does not exist. Walks the filesystem to find TSeries, Suite2p, `.sima`, EEG, and behavior subdirectories.

- **`SessionData` / `load_session()`** (`src.io.session_io`) — convenience loader: given a `.sima` folder and an optional Suite2p folder, returns a `SessionData` with `dfof`, `eeg`, `mobility`, and `velocity` already populated. `velocity` is resampled to the imaging frame grid via `BehaviorData.resample_to_imaging()`; loading falls back to a warning if `imaging_metadata.json` is missing or the velocity file is in the old flat format.

- **`SleepExperiment`** (`src.io.sleep_experiment`) — creates the standard subfolder layout (`behavior/`, `eeg/`, `plots/`) inside `.sima` directories.

- **`Suite2p`** (`src.io.suite2p_io`) — loads Suite2p output (`.npy` files). Prefers the `combined/` directory and falls back to `plane0/`. Key methods: `get_cells()`, `get_npil()`, `get_spikes()`, `get_iscell_indices()`, `plot_time_avg_image()`.

- **`Imaging`** (`src.io.imaging_io`) — reads `imaging_metadata.json`; handles both single-plane and multi-plane frame rate calculations.

- **`BehaviorData`** (`src.io.behavior_io`) — loads and resamples velocity data; computes mobility/immobility epochs. Key methods:
  - `load_processed_velocity(file_name)` → `np.ndarray` — reads `filtered_velocity.json`, which contains `[[time_s, velocity_cms], ...]` pairs (shape `(N, 2)`, **signed cm/s**, positive = forward) as written by `notebooks/behavior/process_velocity.ipynb`.
  - `resample_to_imaging(velocity_ts, imaging_fps, n_frames)` → `np.ndarray shape (n_frames,)` — resamples an event-driven velocity time series onto the imaging frame grid. Replaces `lab3.BehaviorExperiment.format_behavior_data(sampling_interval=frame_period)`. `velocity_ts` must be shape `(N, 2)` with column 0 = seconds and strictly monotonically increasing timestamps. Returns raw resampled **signed velocity in cm/s** (no smoothing); apply Gaussian filter after if needed. Raises `ValueError` for short behavior recordings.
  - `define_mobility(velocity, threshold=IMMOBILITY_VELOCITY_THRESHOLD, ...)` → `pd.Series` of booleans — rolling-window mobility classification. `velocity` must be in **cm/s** (signed); `abs()` is applied internally so backward motion is classified mobile. `threshold` defaults to `config.IMMOBILITY_VELOCITY_THRESHOLD` (1.0 cm/s), matching `brain_state_filter`'s `>=` boundary. Reads imaging fps from `imaging_metadata.json` at the TSeries root (two levels above `behavior_dir`).

- **`EegData`** (`src.io.eeg_io`) — reads scored EEG CSVs (`sleep.csv`). Converts integer scores to brain-state columns: awake=0, NREM=1, REM=2, other=3.

- **`Suite2pDFOF` / `JiaDFOF`** (`src.preprocessing.dfof`) — strategy-pattern dF/F calculators. `Suite2pDFOF` does neuropil-corrected dF/F (coefficient 0.7) using a maxmin filter baseline. `JiaDFOF` follows Jia et al. 2010. Both accept `signal` as a 2D `(n_rois, n_timepoints)` array.

- **`ExperimentDatabase`** (`src.db.database`) — MySQL wrapper via pymysql. Connection parameters from env vars `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`.

## Data Layout on Disk

```
/data2/gergely/invivo_DATA/sleep/
  <mouse_id>/
    TSeries-YYYYMMDD-HHh.../     # raw imaging
      <TSeries>.sima/
        behavior/                # filtered_velocity.json
        eeg/                     # sleep.csv, velo_eeg.csv
        plots/
    suite2p.../
      combined/ or plane0/       # F.npy, Fneu.npy, spks.npy, iscell.npy, stat.npy
```

## Scripts

All scripts configure logging via `logging.basicConfig()` inside their `if __name__ == "__main__":` guard (or at the top of `main()` for scripts imported by tests). Run any script with `--help` for full argument documentation.

### process_tdml_behavior_data.py

`scripts/behavior_scripts/process_tdml_behavior_data.py` converts `.tdml` or `.vr` BehaviorMate files into structured output. By default it writes a `.json` file in the same directory as the input file.

```bash
# single file
python scripts/behavior_scripts/process_tdml_behavior_data.py -f path/to/file.tdml

# entire directory tree (skips files with an existing .json; use -o to overwrite)
python scripts/behavior_scripts/process_tdml_behavior_data.py -d path/to/dir

# write .pkl instead of .json
python scripts/behavior_scripts/process_tdml_behavior_data.py -f file.tdml --file_type pkl

# also load into the SQL database (requires lab3 and DB env vars)
python scripts/behavior_scripts/process_tdml_behavior_data.py -f file.tdml --sql -g my_group
```

The `lab3` database dependency is imported lazily inside `loadSql` — the script runs without `lab3` installed as long as `--sql` is not passed.

### Behavior preprocessing pipeline

Raw BehaviorMate recordings flow through three steps before analysis:

```
raw .tdml  (event-driven, timestamped)
  → scripts/behavior_scripts/process_tdml_behavior_data.py
      writes <session>.json alongside the .tdml
      key output keys:
        "treadmillPosition": [[time_s, norm_pos], ...]   (normalized, wraps at 1)
        "treadmillDy":       [[time_s, dy_counts], ...]  (raw encoder increments, wrap-immune)
        "position_scale":    -2.15                        (counts/mm; negative=forward is neg dy)

  → notebooks/behavior/process_velocity.ipynb
      computes signed velocity from dy (NOT from differencing normalized position):
        distance_mm  = dy[i] / position_scale
        velocity_cms = (distance_mm / 10.0) / dt
        t_stamp      = time[i-1]   (start-of-interval alignment)
      optional: drop outlier events by index range
      writes <TSeries>.sima/behavior/filtered_velocity.json:
        [[time_s, velocity_cms], ...]  — signed cm/s, positive = forward
      Gaussian smoothing is shown for inspection only — NOT saved

  → BehaviorData.resample_to_imaging(velocity_ts, imaging_fps, n_frames)
      linear interpolation onto uniform imaging-rate grid (np.interp)
      called automatically by load_session(); apply Gaussian filter after
```

The notebook does **not** Gaussian-filter before saving. Smoothing (`scipy.ndimage.gaussian_filter`) belongs in the analysis notebook, after resampling to the imaging grid where σ in samples has a consistent physical meaning.

**Unit change (breaking):** `filtered_velocity.json` now stores signed cm/s (~200× larger than the previous normalised pos/s). Consumers reading it directly must extract column 1 (`arr[:, 1]`) and apply `abs()` when speed rather than signed velocity is needed.

### Known-broken / deferred

These consumers are broken by the format change and require fixes in a separate scope:

- **`scripts/eeg_scripts/eeg_velocity_processing.py:70-71`** — `pd.DataFrame(velocity_json, columns=["velocity"])` crashes on 2-column `[[time_s, vel], ...]` data (`ValueError: 1 columns passed, 2 in data`). Fix: extract column 1 and rename column to `"velocity_cms"` before concat with EEG.
- **`scripts/behavior_scripts/export_mobility_immobility.py:39-40`** — passes raw `(N, 2)` timestamped array to `define_mobility`, which expects a 1D per-frame velocity (after `resample_to_imaging`). Needs rewrite to use `load_session()` or an explicit resample step. **Flagged as the immediate next task.**
- **`notebooks/behavior/behavior_mobility_immobility.ipynb` cell 4** — same class as above (old calling convention). Needs rewrite alongside `export_mobility_immobility.py`.

### Other scripts

- **`add_project_subfolders.py`** — takes a `mouse_id` and creates the standard `behavior/`, `eeg/`, `plots/` subfolder layout inside every `.sima` directory for that mouse.
- **`add_time_avg.py`** — walks a directory tree, finds `suite2p/` folders, and saves a time-averaged image (PNG by default; `--tif` for multipage TIFF).
- **`behavior_scripts/export_mobility_immobility.py`** — takes a `mouse_id`, loads velocity data for each behavior folder, and writes `mobility_immobility.json` alongside the velocity file.
- **`eeg_scripts/eeg_velocity_processing.py`** — merges a scored EEG CSV with velocity data, resampling if lengths differ, and writes `velo_eeg.csv` to the EEG folder.
- **`eeg_scripts/process_matlab_files.py`** — extracts the `sleepData/state` array from a MATLAB `.mat` file and writes it as `sleep.csv`.

## Logging

**Convention for every `src/` module:**

```python
import logging
# ... other imports ...
logger = logging.getLogger(__name__)   # module-level, after all imports
```

All log calls use `logger.*`, never `logging.*` directly. Use lazy %-style args:

```python
logger.warning("No .sima folder found in %s", path)   # correct
logger.warning(f"No .sima folder found in {path}")    # wrong — always evaluated
```

Inside `except` blocks use `logger.exception(...)` when the traceback is useful, `logger.error(...)` otherwise.

**Hard rules for library modules (`src/`):**
- Never call `logging.basicConfig()`, `addHandler()`, or `setLevel()` — those belong in entry points (scripts, notebooks), not library code.
- Never log credentials or env-var values (`DB_HOST`, `DB_USER`, `DB_PASSWORD`, etc.).

**Convention for scripts (`scripts/`):**

Scripts are entry points and configure logging themselves:

```python
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
```

Use `logging.*` directly (no named logger required). `print()` is fine for genuine CLI progress output, but error handling in `except` blocks should use `logging.error()` / `logging.exception()` rather than `print()`. Lazy %-style args apply here too.

`src/__init__.py` installs a `NullHandler` on the `src` package so library log calls are silenced by default when no entry point has configured logging.

## Google Colab vs. Local

`src/colab/google_utils.py` and `src/colab/google_drive.py` import `google.colab` and will fail locally — both files guard this with a `try/except ImportError`. Everything else in `src/` is usable in either environment.
