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

The package is installed as `sleep` (editable). Additional pip dependencies not in `environment.yaml` are listed in `requirements.txt`; install both.

## Running Notebooks

```bash
jupyter notebook
```

Tests are also notebooks (in `tests/`), not a pytest suite. Open them in Jupyter and run cells manually. There are no shell-runnable tests.

## Project Structure

```
src/
  classes/         # Core dataclasses and analysis classes
  xcorr/           # Cross-correlation analysis
  *.py             # Standalone analysis modules
notebooks/         # Analysis organized by topic
scripts/           # One-off processing scripts
tests/             # Notebook-based tests
```

### Key Classes (`src/classes/`)

- **`MouseData`** (`mouse_class.py`) — entry point for a mouse's data. Takes `mouse_id` and `root_folder` (default `/data2/gergely/invivo_DATA/sleep`). Walks the filesystem to find TSeries, Suite2p, `.sima`, EEG, and behavior subdirectories.

- **`SleepExperiment`** (`sleep_experiment.py`) — wraps a single TSeries recording session. Creates the standard subfolder layout (`behavior/`, `eeg/`, `plots/`) inside `.sima` directories.

- **`Suite2p`** (`suite2p_class.py`) — loads Suite2p output (`.npy` files). Prefers the `combined/` directory and falls back to `plane0/` for single-plane data. Key methods: `get_cells()`, `get_npil()`, `get_spikes()`, `get_iscell_indices()`, `plot_time_avg_image()`.

- **`Imaging`** (`imaging_class.py`) — reads `imaging_metadata.json` from a TSeries folder; handles both single-plane and multi-plane frame rate calculations.

- **`BehaviorData`** (`behavior_class.py`) — loads `filtered_velocity.json` and computes mobility/immobility epochs. Requires `imaging_metadata.json` to exist two levels above the behavior directory (TSeries root) to get the frame rate.

- **`EegData`** (`eeg_class.py`) — reads scored EEG CSVs (`sleep.csv`). Converts integer scores to brain-state columns: awake=0, NREM=1, REM=2, other=3. Also loads `velo_eeg.csv` for velocity-EEG combined analysis.

- **`Suite2pDFOF` / `JiaDFOF`** (`dfof.py`) — strategy-pattern dF/F calculators. `Suite2pDFOF` does neuropil-corrected dF/F using a maxmin filter baseline. `JiaDFOF` follows Jia et al. 2010. Both accept `signal` as a 2D `(n_rois, n_timepoints)` array.

- **`ExperimentDatabase`** (`database.py`) — MySQL wrapper via pymysql. Connection parameters come from env vars `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`.

### Standalone Modules (`src/`)

- `frequency_psd.py` — Welch PSD and frequency spectrum utilities
- `full_frame_analysis.py` — Butterworth filters and autocorrelation for full-frame signals
- `clustering.py` — time-series clustering helpers
- `custom_statistics.py` / `estimation_stats.py` — statistical utilities (includes dabest integration)
- `google_drive.py` — Google Drive/Sheets utilities (Colab-only; uses `google.colab.auth`)
- `xcorr/xcorr_analysis.py` — cross-correlation and heatmap plotting

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

## Google Colab vs. Local

`src/classes/google_utils.py` and `src/google_drive.py` import `google.colab` and will fail locally. These are Colab-only utilities. Everything else in `src/` is usable in either environment.
