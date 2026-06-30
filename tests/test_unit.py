"""Unit tests for src/ modules.

Run with:
    pip install pytest
    pytest tests/test_unit.py -v

Tests that require data on disk are marked skip. All other tests use only
synthetic or in-memory data and should pass without any data files.
"""

import importlib.util
import json
import pickle
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

class TestConfig:
    def test_numeric_constants_positive(self):
        from src.config import (
            MAXMIN_WINDOW, MAXMIN_SIGMA, FILTER_MIN_PERIODS,
            SUITE2P_DFOF_WINDOW, SUITE2P_DFOF_SIGMA, SUITE2P_DFOF_MIN_PERIODS,
            JIA_T1, JIA_T2, MIN_EPOCH_FRAMES, MIN_SLEEP_EPOCH_FRAMES,
            MAX_AUTOCORR_LAGS, NPIL_COEFF,
        )
        for name, val in [
            ("MAXMIN_WINDOW", MAXMIN_WINDOW),
            ("MAXMIN_SIGMA", MAXMIN_SIGMA),
            ("SUITE2P_DFOF_WINDOW", SUITE2P_DFOF_WINDOW),
            ("SUITE2P_DFOF_SIGMA", SUITE2P_DFOF_SIGMA),
            ("JIA_T1", JIA_T1),
            ("JIA_T2", JIA_T2),
            ("MIN_EPOCH_FRAMES", MIN_EPOCH_FRAMES),
            ("MIN_SLEEP_EPOCH_FRAMES", MIN_SLEEP_EPOCH_FRAMES),
            ("MAX_AUTOCORR_LAGS", MAX_AUTOCORR_LAGS),
            ("NPIL_COEFF", NPIL_COEFF),
        ]:
            assert val > 0, f"{name} should be positive"

    def test_min_periods_fractions(self):
        from src.config import FILTER_MIN_PERIODS, SUITE2P_DFOF_MIN_PERIODS
        assert 0 < FILTER_MIN_PERIODS <= 1
        assert 0 < SUITE2P_DFOF_MIN_PERIODS <= 1

    def test_known_eeg_codes(self):
        from src.config import KNOWN_EEG_CODES
        assert KNOWN_EEG_CODES == frozenset({0, 1, 2, 3})

    def test_immobility_threshold_positive(self):
        from src.config import IMMOBILITY_VELOCITY_THRESHOLD
        assert IMMOBILITY_VELOCITY_THRESHOLD > 0


# ---------------------------------------------------------------------------
# preprocessing/filters.py
# ---------------------------------------------------------------------------

class TestMaxminFilter:
    def _signal(self, n_rois=3, n_frames=200):
        rng = np.random.default_rng(0)
        return rng.normal(loc=1.0, scale=0.1, size=(n_rois, n_frames))

    def test_output_shape(self):
        from src.preprocessing.filters import maxmin_filter
        sig = self._signal()
        out = maxmin_filter(sig, window=20, sigma=2, min_periods=0.2)
        assert out.shape == sig.shape

    def test_output_is_dataframe(self):
        from src.preprocessing.filters import maxmin_filter
        sig = self._signal()
        out = maxmin_filter(sig, window=20, sigma=2, min_periods=0.2)
        assert isinstance(out, pd.DataFrame)

    def test_baseline_not_above_signal_max(self):
        """The rolling-max-of-rolling-min baseline should not exceed the signal maximum."""
        from src.preprocessing.filters import maxmin_filter
        sig = self._signal()
        out = maxmin_filter(sig, window=20, sigma=2, min_periods=0.2)
        assert float(out.max().max()) <= float(np.max(sig)) + 1e-6


# ---------------------------------------------------------------------------
# preprocessing/dfof.py — DFOFStrategy base
# ---------------------------------------------------------------------------

class TestDFOFStrategyValidation:
    def test_1d_signal_raises(self):
        from src.preprocessing.dfof import JiaDFOF
        strategy = JiaDFOF(t1=5, t2=20)
        with pytest.raises(ValueError, match="2D"):
            strategy.calculate(signal=np.ones(50))


# ---------------------------------------------------------------------------
# preprocessing/dfof.py — JiaDFOF
# ---------------------------------------------------------------------------

class TestJiaDFOF:
    def _signal(self, n_rois=4, n_frames=300):
        rng = np.random.default_rng(1)
        return rng.normal(loc=100.0, scale=5.0, size=(n_rois, n_frames))

    def test_output_shape(self):
        from src.preprocessing.dfof import JiaDFOF
        sig = self._signal()
        out = JiaDFOF(t1=10, t2=50).calculate(signal=sig)
        assert out.shape == sig.shape

    def test_output_is_dataframe(self):
        from src.preprocessing.dfof import JiaDFOF
        sig = self._signal()
        out = JiaDFOF(t1=10, t2=50).calculate(signal=sig)
        assert isinstance(out, pd.DataFrame)

    def test_slow_trend_subtraction(self):
        """With slow_trend_window set, output should differ from without."""
        from src.preprocessing.dfof import JiaDFOF
        sig = self._signal()
        out_no_trend = JiaDFOF(t1=10, t2=50).calculate(signal=sig)
        out_trend = JiaDFOF(t1=10, t2=50, slow_trend_window=100).calculate(signal=sig)
        assert not np.allclose(out_no_trend.values, out_trend.values, equal_nan=True)


# ---------------------------------------------------------------------------
# preprocessing/dfof.py — Suite2pDFOF
# ---------------------------------------------------------------------------

class TestSuite2pDFOF:
    def _arrays(self, n_rois=3, n_frames=200):
        rng = np.random.default_rng(2)
        signal = rng.normal(loc=500.0, scale=20.0, size=(n_rois, n_frames))
        npil = rng.normal(loc=200.0, scale=10.0, size=(n_rois, n_frames))
        return signal, npil

    def test_output_shape(self):
        from src.preprocessing.dfof import Suite2pDFOF
        sig, npil = self._arrays()
        out = Suite2pDFOF(window=20, sigma=2).calculate(signal=sig, npil=npil)
        assert out.shape == sig.shape

    def test_custom_npil_coeff(self):
        """Changing npil_coeff should produce a different result."""
        from src.preprocessing.dfof import Suite2pDFOF
        sig, npil = self._arrays()
        out_07 = Suite2pDFOF(window=20, sigma=2, npil_coeff=0.7).calculate(signal=sig, npil=npil)
        out_05 = Suite2pDFOF(window=20, sigma=2, npil_coeff=0.5).calculate(signal=sig, npil=npil)
        assert not np.allclose(out_07.values, out_05.values, equal_nan=True)

    def test_1d_signal_raises(self):
        from src.preprocessing.dfof import Suite2pDFOF
        with pytest.raises(ValueError, match="2D"):
            Suite2pDFOF().calculate(signal=np.ones(50), npil=np.ones(50))

    def test_baseline_cached_then_cleared(self):
        """Each call to calculate() should recompute a fresh baseline."""
        from src.preprocessing.dfof import Suite2pDFOF
        sig, npil = self._arrays()
        strategy = Suite2pDFOF(window=20, sigma=2)
        out1 = strategy.calculate(signal=sig, npil=npil)
        out2 = strategy.calculate(signal=sig * 1.1, npil=npil)
        assert not np.allclose(out1.values, out2.values, equal_nan=True)


# ---------------------------------------------------------------------------
# io/eeg_io.py — EegData
# ---------------------------------------------------------------------------

class TestEegData:
    def _write_csv(self, tmp_path, content, filename="sleep.csv"):
        p = tmp_path / filename
        p.write_text(content)
        return tmp_path

    def test_two_column_csv(self, tmp_path):
        from src.io.eeg_io import EegData
        self._write_csv(tmp_path, "time,score\n0,0\n1,1\n2,2\n3,3\n")
        df = EegData(tmp_path).import_scored_eeg()
        assert list(df.columns) == ["time", "score", "awake", "NREM", "REM", "other"]
        assert df["NREM"].iloc[1] == 1
        assert df["REM"].iloc[2] == 1

    def test_one_column_csv(self, tmp_path):
        from src.io.eeg_io import EegData
        self._write_csv(tmp_path, "score\n0\n1\n2\n")
        df = EegData(tmp_path).import_scored_eeg()
        assert "time" in df.columns
        assert "NREM" in df.columns

    def test_unknown_score_row_has_zero_states(self, tmp_path):
        """Unknown EEG score code → all brain-state columns are 0 for that row."""
        from src.io.eeg_io import EegData
        self._write_csv(tmp_path, "score\n0\n1\n99\n")
        df = EegData(tmp_path).import_scored_eeg()
        row = df[df["score"] == 99].iloc[0]
        assert row["awake"] == 0 and row["NREM"] == 0 and row["REM"] == 0 and row["other"] == 0

    def test_three_column_csv_raises(self, tmp_path):
        from src.io.eeg_io import EegData
        self._write_csv(tmp_path, "a,b,c\n0,1,2\n")
        with pytest.raises(ValueError, match="Unexpected number of columns"):
            EegData(tmp_path).import_scored_eeg()

    def test_missing_file_raises(self, tmp_path):
        from src.io.eeg_io import EegData
        with pytest.raises(FileNotFoundError):
            EegData(tmp_path).import_scored_eeg()


# ---------------------------------------------------------------------------
# io/imaging_io.py — Imaging
# ---------------------------------------------------------------------------

class TestImaging:
    def test_get_metadata_single_plane(self, tmp_path):
        from src.io.imaging_io import Imaging
        meta = {"sequence_type": "single plane", "fps": 10.0}
        (tmp_path / "imaging_metadata.json").write_text(json.dumps(meta))
        loaded = Imaging(tmp_path).get_imaging_metadata()
        assert loaded["fps"] == 10.0

    def test_missing_metadata_raises(self, tmp_path):
        from src.io.imaging_io import Imaging
        with pytest.raises(FileNotFoundError):
            Imaging(tmp_path).get_imaging_metadata()

    def test_corrupt_metadata_raises(self, tmp_path):
        from src.io.imaging_io import Imaging
        (tmp_path / "imaging_metadata.json").write_text("not json{{{")
        with pytest.raises(ValueError, match="Could not decode JSON"):
            Imaging(tmp_path).get_imaging_metadata()

    def test_multiplane_frame_rate(self, tmp_path):
        from src.io.imaging_io import Imaging
        meta = {
            "sequence_type": "multi plane",
            "number of images/plane": "600",
            "recording length in seconds": "60.0",
        }
        (tmp_path / "imaging_metadata.json").write_text(json.dumps(meta))
        fps = Imaging(tmp_path).multiplane_frame_rate()
        assert fps == pytest.approx(10.0, rel=1e-3)


# ---------------------------------------------------------------------------
# analysis/calc_module.py — label_consecutive_states
# ---------------------------------------------------------------------------

class TestLabelConsecutiveStates:
    def test_no_long_runs_returns_all_false(self):
        from src.analysis.calc_module import label_consecutive_states
        data = pd.Series([True] * 50 + [False] * 50)
        result = label_consecutive_states(data, "NREM")
        assert (result == False).all()

    def test_long_run_gets_labeled(self):
        from src.analysis.calc_module import label_consecutive_states
        data = pd.Series([True] * 110)
        result = label_consecutive_states(data, "NREM")
        labeled = result[result != False]
        assert len(labeled) > 0
        assert all(str(v).startswith("NREM") for v in labeled)


# ---------------------------------------------------------------------------
# db/database.py — _project_filter_sql validation
# ---------------------------------------------------------------------------

class TestProjectFilterSql:
    def test_valid_name_does_not_raise(self):
        from src.db.database import _project_filter_sql
        sql = _project_filter_sql("sleep_project-1")
        assert "sleep_project-1" in sql

    def test_sql_metachar_raises(self):
        from src.db.database import _project_filter_sql
        with pytest.raises(ValueError, match="invalid characters"):
            _project_filter_sql("'; DROP TABLE mice; --")

    def test_semicolon_raises(self):
        from src.db.database import _project_filter_sql
        with pytest.raises(ValueError, match="invalid characters"):
            _project_filter_sql("name; DROP TABLE")


# ---------------------------------------------------------------------------
# scripts/behavior_scripts/process_tdml_behavior_data.py
# ---------------------------------------------------------------------------

class TestProcessTdmlBehaviorData:
    """Tests for the TDML → JSON processing script."""

    _TDML = Path(__file__).parent / "behavior/140302_1_20231211123734.tdml"

    @pytest.fixture
    def mod(self):
        """Load the script as a module without executing __main__."""
        spec = importlib.util.spec_from_file_location(
            "process_tdml_behavior_data",
            Path(__file__).parent.parent
            / "scripts/behavior_scripts/process_tdml_behavior_data.py",
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    @pytest.fixture
    def tdml(self, tmp_path):
        """Copy the test .tdml into a temp dir so output files don't pollute the repo."""
        dst = tmp_path / self._TDML.name
        shutil.copy(self._TDML, dst)
        return dst

    # --- numpy_to_list ---

    def test_numpy_to_list_array(self, mod):
        result = mod.numpy_to_list(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_numpy_to_list_dict_with_array_values(self, mod):
        result = mod.numpy_to_list({"a": np.array([1, 2]), "b": 3})
        assert isinstance(result["a"], list)
        assert result["b"] == 3

    def test_numpy_to_list_nested(self, mod):
        result = mod.numpy_to_list([np.array([1, 2]), [np.array([3, 4])]])
        assert result == [[1, 2], [[3, 4]]]

    def test_numpy_to_list_scalar_passthrough(self, mod):
        assert mod.numpy_to_list(42) == 42
        assert mod.numpy_to_list("hello") == "hello"

    # --- findFiles ---

    def test_find_files_finds_tdml(self, mod, tmp_path):
        (tmp_path / "run.tdml").write_text("")
        found = list(mod.findFiles(str(tmp_path)))
        assert any("run.tdml" in f for f in found)

    def test_find_files_finds_vr(self, mod, tmp_path):
        (tmp_path / "run.vr").write_text("")
        found = list(mod.findFiles(str(tmp_path)))
        assert any("run.vr" in f for f in found)

    def test_find_files_skips_when_json_exists(self, mod, tmp_path):
        (tmp_path / "run.tdml").write_text("")
        (tmp_path / "run.json").write_text("{}")
        found = list(mod.findFiles(str(tmp_path), overwrite=False))
        assert found == []

    def test_find_files_overwrite_ignores_existing_json(self, mod, tmp_path):
        (tmp_path / "run.tdml").write_text("")
        (tmp_path / "run.json").write_text("{}")
        found = list(mod.findFiles(str(tmp_path), overwrite=True))
        assert len(found) == 1

    # --- main: output format ---

    def test_main_writes_json_next_to_tdml(self, mod, tdml):
        mod.main(["-f", str(tdml)])
        assert tdml.with_suffix(".json").exists()

    def test_main_json_has_expected_keys(self, mod, tdml):
        mod.main(["-f", str(tdml)])
        data = json.loads(tdml.with_suffix(".json").read_text())
        for key in ("trackLength", "recordingDuration", "treadmillPosition"):
            assert key in data, f"missing key: {key}"

    def test_main_does_not_write_pkl_by_default(self, mod, tdml):
        mod.main(["-f", str(tdml)])
        assert not tdml.with_suffix(".pkl").exists()

    def test_main_file_type_pkl_writes_pkl(self, mod, tdml):
        mod.main(["-f", str(tdml), "--file_type", "pkl"])
        pkl = tdml.with_suffix(".pkl")
        assert pkl.exists()
        data = pickle.loads(pkl.read_bytes())
        assert "treadmillPosition" in data

    # --- main: database opt-in ---

    def test_main_does_not_call_load_sql_by_default(self, mod, tdml, monkeypatch):
        called = []
        monkeypatch.setattr(mod, "loadSql", lambda *a, **kw: called.append(True))
        mod.main(["-f", str(tdml)])
        assert called == []

    def test_main_trial_id_without_sql_logs_warning(self, mod, tdml, caplog):
        mod.main(["-f", str(tdml), "--trial_id", "42"])
        assert any("--trial_id" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# io/behavior_io.py — BehaviorData.resample_to_imaging
# ---------------------------------------------------------------------------

class TestResampleToImaging:
    """Tests for BehaviorData.resample_to_imaging().

    All tests use synthetic or sample-file-derived data; no live data required.
    The sample JSON is the output of process_tdml_behavior_data.py on the
    accompanying .tdml fixture.
    """

    _SAMPLE_JSON = Path(__file__).parent / "behavior/140302_1_20231211123734.json"

    @pytest.fixture
    def bd(self, tmp_path):
        from src.io.behavior_io import BehaviorData
        return BehaviorData(tmp_path)

    @pytest.fixture
    def sample_velocity_ts(self):
        """Derive velocity_ts [[time_s, velocity], ...] from sample treadmillPosition.

        Velocity at each event = Δnormalized_position / Δtime_s.
        Timestamps are the start of each inter-event interval.
        This mirrors what the reworked process_velocity notebook will write.
        """
        with open(self._SAMPLE_JSON) as f:
            tp = json.load(f)["treadmillPosition"]
        tp = np.array(tp)
        dt = np.diff(tp[:, 0])
        dp = np.diff(tp[:, 1])
        times = tp[:-1, 0]
        velocity = dp / dt
        return np.column_stack([times, velocity])

    # --- shape and argument validation ---

    def test_invalid_shape_1d_raises(self, bd):
        with pytest.raises(ValueError, match="shape"):
            bd.resample_to_imaging(np.ones(10), imaging_fps=15.0, n_frames=5)

    def test_invalid_shape_wrong_cols_raises(self, bd):
        with pytest.raises(ValueError, match="shape"):
            bd.resample_to_imaging(np.ones((10, 3)), imaging_fps=15.0, n_frames=5)

    def test_nonpositive_fps_raises(self, bd):
        vts = np.column_stack([np.array([0.0, 1.0]), np.ones(2)])
        with pytest.raises(ValueError, match="imaging_fps"):
            bd.resample_to_imaging(vts, imaging_fps=0.0, n_frames=5)

    def test_nonpositive_n_frames_raises(self, bd):
        vts = np.column_stack([np.array([0.0, 1.0]), np.ones(2)])
        with pytest.raises(ValueError, match="n_frames"):
            bd.resample_to_imaging(vts, imaging_fps=15.0, n_frames=0)

    # --- monotonicity guard ---

    def test_non_monotonic_timestamps_raises(self, bd):
        # t=0, 1, 0.5 (reversal) → not strictly increasing
        t = np.array([0.0, 1.0, 0.5, 2.0])
        vts = np.column_stack([t, np.zeros(4)])
        with pytest.raises(ValueError, match="strictly increasing"):
            bd.resample_to_imaging(vts, imaging_fps=1.0, n_frames=2)

    def test_duplicate_timestamps_raise(self, bd):
        # zero diff also violates strict monotonicity
        t = np.array([0.0, 1.0, 1.0, 2.0])
        vts = np.column_stack([t, np.zeros(4)])
        with pytest.raises(ValueError, match="strictly increasing"):
            bd.resample_to_imaging(vts, imaging_fps=1.0, n_frames=2)

    # --- behavior shorter than imaging ---

    def test_behavior_shorter_than_imaging_raises(self, bd):
        # behavior covers 5 s; imaging needs 10 s (100 frames at 10 fps)
        t = np.linspace(0.0, 5.0, 50)
        vts = np.column_stack([t, np.ones(50)])
        with pytest.raises(ValueError, match="Behavior ends"):
            bd.resample_to_imaging(vts, imaging_fps=10.0, n_frames=100)

    def test_behavior_shorter_error_names_gap(self, bd):
        # error message should quantify the gap so callers know how bad it is
        t = np.linspace(0.0, 5.0, 50)
        vts = np.column_stack([t, np.ones(50)])
        with pytest.raises(ValueError, match=r"\d+\.\d+ s"):
            bd.resample_to_imaging(vts, imaging_fps=10.0, n_frames=100)

    # --- unit correctness (known analytic values) ---

    def test_linear_interpolation_values(self, bd):
        # Two events: (0s, 5.0) and (1s, 10.0) → linear ramp
        # fps=2 → target = [0.0, 0.5, 1.0, 1.5] s
        # expected:       [5.0, 7.5, 10.0, 10.0]  (np.interp clamps past last)
        vts = np.array([[0.0, 5.0], [1.0, 10.0]])
        result = bd.resample_to_imaging(vts, imaging_fps=2.0, n_frames=4)
        np.testing.assert_allclose(result, [5.0, 7.5, 10.0, 10.0])

    def test_constant_velocity_preserved(self, bd):
        t = np.array([0.0, 5.0, 10.0])
        vts = np.column_stack([t, np.full(3, 3.0)])
        result = bd.resample_to_imaging(vts, imaging_fps=1.0, n_frames=10)
        np.testing.assert_allclose(result, np.full(10, 3.0))

    # --- output length is exactly n_frames ---

    def test_output_length_equals_n_frames(self, bd):
        t = np.linspace(0.0, 100.0, 500)
        vts = np.column_stack([t, np.random.default_rng(0).normal(size=500)])
        result = bd.resample_to_imaging(vts, imaging_fps=10.0, n_frames=200)
        assert result.shape == (200,)

    # --- behavior span longer than imaging (normal trim, implicit) ---

    def test_behavior_longer_than_imaging_no_error(self, bd, sample_velocity_ts):
        # sample spans ~3598 s; imaging ends at ~333 s (5000 frames at 15 fps)
        result = bd.resample_to_imaging(
            sample_velocity_ts, imaging_fps=15.0, n_frames=5000
        )
        assert result.shape == (5000,)

    # --- effective upsampling ---

    def test_upsampling_output_length_and_values(self, bd):
        # 6 sparse events over 10 s; 100 frames at 10 fps
        t = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        v = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        vts = np.column_stack([t, v])
        result = bd.resample_to_imaging(vts, imaging_fps=10.0, n_frames=100)
        assert result.shape == (100,)
        # frame 20 → t=2.0 s → v should be 2.0 (exact event)
        assert result[20] == pytest.approx(2.0)
        # frame 10 → t=1.0 s → midpoint between (0,0) and (2,2) → 1.0
        assert result[10] == pytest.approx(1.0)

    # --- effective downsampling ---

    def test_downsampling_output_length_and_values(self, bd):
        # 1000 dense events over 10 s; 20 frames at 2 fps
        t = np.linspace(0.0, 10.0, 1000)
        v = t.copy()  # velocity == time for easy checking
        vts = np.column_stack([t, v])
        result = bd.resample_to_imaging(vts, imaging_fps=2.0, n_frames=20)
        assert result.shape == (20,)
        # frame 0 → t=0 → v=0; frame 10 → t=5 → v=5
        assert result[0] == pytest.approx(0.0, abs=0.01)
        assert result[10] == pytest.approx(5.0, abs=0.01)

    # --- spans roughly equal ---

    def test_spans_roughly_equal(self, bd):
        # behavior ends within one frame of last imaging frame
        fps = 15.0
        n_frames = 100
        last_frame_time = (n_frames - 1) / fps  # 6.6 s
        # behavior ends at last_frame_time + 0.5 frames (within tolerance)
        t_end = last_frame_time + 0.5 / fps
        t = np.array([0.0, t_end / 2, t_end])
        vts = np.column_stack([t, np.ones(3)])
        result = bd.resample_to_imaging(vts, imaging_fps=fps, n_frames=n_frames)
        assert result.shape == (n_frames,)

    # --- return type ---

    def test_returns_ndarray(self, bd):
        vts = np.array([[0.0, 1.0], [10.0, 2.0]])
        result = bd.resample_to_imaging(vts, imaging_fps=1.0, n_frames=5)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Stubs requiring data on disk
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="requires Suite2p .npy output files on disk")
def test_suite2p_get_cells():
    from src.io.suite2p_io import Suite2p
    s2p = Suite2p("/data2/gergely/invivo_DATA/sleep/example/suite2p")
    cells = s2p.get_cells()
    assert cells.ndim == 2


@pytest.mark.skip(reason="requires TSeries folder with imaging_metadata.json and filtered_velocity.json")
def test_behavior_define_mobility():
    from src.io.behavior_io import BehaviorData
    behavior_dir = Path("/data2/gergely/invivo_DATA/sleep/example/TSeries.sima/behavior")
    bd = BehaviorData(behavior_dir)
    velocity = bd.load_processed_velocity()
    mobility = bd.define_mobility(velocity)
    assert len(mobility) == len(velocity)


@pytest.mark.skip(reason="requires live MySQL database connection")
def test_database_connection():
    from src.db.database import ExperimentDatabase
    db = ExperimentDatabase()
    result = db.select_all("SELECT 1 AS ping")
    assert result[0]["ping"] == 1
    db.disconnect()
