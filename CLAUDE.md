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
  config.py              # Centralized analysis parameters (all numeric defaults)
  logging_setup.py       # Logger configuration helper
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
scripts/
  behavior_scripts/
    process_json_behavior_data.py  # Parses .tdml/.vr files → .json (default) or .pkl
tests/
  behavior/              # Sample .tdml files used by pytest integration tests
  test_unit.py           # pytest suite (includes TestProcessJsonBehaviorData)
```

### Key classes and their import paths

- **`MouseData`** (`src.io.mouse_io`) — entry point for a mouse's data. Takes `mouse_id` and `root_folder` (env var `SLEEP_DATA_ROOT`, default `/data2/gergely/invivo_DATA/sleep`). Walks the filesystem to find TSeries, Suite2p, `.sima`, EEG, and behavior subdirectories.

- **`SessionData` / `load_session()`** (`src.io.session_io`) — convenience loader: given a `.sima` folder and an optional Suite2p folder, returns a `SessionData` with `dfof`, `eeg`, `mobility`, and `velocity` already populated.

- **`SleepExperiment`** (`src.io.sleep_experiment`) — creates the standard subfolder layout (`behavior/`, `eeg/`, `plots/`) inside `.sima` directories.

- **`Suite2p`** (`src.io.suite2p_io`) — loads Suite2p output (`.npy` files). Prefers the `combined/` directory and falls back to `plane0/`. Key methods: `get_cells()`, `get_npil()`, `get_spikes()`, `get_iscell_indices()`, `plot_time_avg_image()`.

- **`Imaging`** (`src.io.imaging_io`) — reads `imaging_metadata.json`; handles both single-plane and multi-plane frame rate calculations.

- **`BehaviorData`** (`src.io.behavior_io`) — loads `filtered_velocity.json` and computes mobility/immobility epochs. Requires `imaging_metadata.json` two levels above the behavior directory (TSeries root).

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

## Behavior Processing Script

`scripts/behavior_scripts/process_json_behavior_data.py` converts `.tdml` or `.vr` BehaviorMate files into structured output. By default it writes a `.json` file in the same directory as the input file.

```bash
# single file
python scripts/behavior_scripts/process_json_behavior_data.py -f path/to/file.tdml

# entire directory tree (skips files with an existing .json; use -o to overwrite)
python scripts/behavior_scripts/process_json_behavior_data.py -d path/to/dir

# write .pkl instead of .json
python scripts/behavior_scripts/process_json_behavior_data.py -f file.tdml --file_type pkl

# also load into the SQL database (requires lab3 and DB env vars)
python scripts/behavior_scripts/process_json_behavior_data.py -f file.tdml --sql -g my_group
```

The `lab3` database dependency is imported lazily inside `loadSql` — the script runs without `lab3` installed as long as `--sql` is not passed.

## Logging

`src/logging_setup.py` is the sole place that configures handlers. Call it from entry points (scripts, notebooks); never from library code.

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
- Never call `logging.basicConfig()`, `addHandler()`, or `setLevel()` — those belong only in `logging_setup.py`.
- Never log credentials or env-var values (`DB_HOST`, `DB_USER`, `DB_PASSWORD`, etc.).

`src/__init__.py` installs a `NullHandler` on the `src` package so library log calls are silenced by default when no entry point has configured logging.

## Google Colab vs. Local

`src/colab/google_utils.py` and `src/colab/google_drive.py` import `google.colab` and will fail locally — both files guard this with a `try/except ImportError`. Everything else in `src/` is usable in either environment.
