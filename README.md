# Core functions for analyzing and visualizing data for the sleep project

The codebase contains a mix of functions and classes that work either locally in the Conda environment provided by the `environment.yaml` file or in Google Colab notebooks.

## Installation

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- Python 3.9+

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/GergelyTuri/sleep.git
   cd sleep
   ```

2. **Create and activate the Conda environment**

   ```bash
   conda env create -f environment.yaml
   conda activate sleep
   ```

3. **Install the package in editable mode**

   ```bash
   pip install -e .
   ```

4. **Install pip-only extras**

   ```bash
   pip install -r requirements.txt
   ```

### Google Colab

The modules under `src/colab/` (`google_utils.py`, `google_drive.py`) require `google.colab` and are only usable inside a Colab notebook. They are guarded with a `try/except ImportError` so they won't break local imports. If you need Google Drive integration in Colab, install the optional dependencies manually:

```bash
pip install gspread google-auth
```

## Running Notebooks

```bash
jupyter notebook
```

## Running Tests

```bash
pytest tests/test_unit.py -v
```

## Scripts

### Behavior data processing

`scripts/behavior_scripts/process_tdml_behavior_data.py` converts BehaviorMate `.tdml` or `.vr` files into structured `.json` output (or `.pkl` if needed). The output file is written to the same directory as the input.

```bash
# single file
python scripts/behavior_scripts/process_tdml_behavior_data.py -f path/to/file.tdml

# entire directory tree (skips files that already have a .json; use -o to overwrite)
python scripts/behavior_scripts/process_tdml_behavior_data.py -d path/to/dir

# write .pkl instead of .json
python scripts/behavior_scripts/process_tdml_behavior_data.py -f file.tdml --file_type pkl

# also load into the SQL database (requires lab3 and DB environment variables)
python scripts/behavior_scripts/process_tdml_behavior_data.py -f file.tdml --sql -g my_group
```
