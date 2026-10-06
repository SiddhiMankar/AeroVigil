"""
Tests for ingestion/cmapss_loader.py

Tests verify:
    1. Dataset directory and file existence detection (error messages).
    2. Correct column assignment after loading.
    3. Column count validation.
    4. RUL label computation: max_cycle - current_cycle.
    5. RUL cap behaviour.
    6. Ground-truth RUL loading.
    7. Engine-level RUL boundary conditions.
    8. No cross-engine RUL leakage.
    9. Test DataFrame has no 'rul' column.
   10. Loader summary statistics.
"""

import os
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure repo root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.cmapss_loader import CMAPSSLoader, load_fd001, _BASE_COLUMNS, _EXPECTED_N_COLS


# ---------------------------------------------------------------------------
# Helpers: Build minimal synthetic C-MAPSS files for unit tests
# ---------------------------------------------------------------------------

def _make_cmapss_row(unit: int, cycle: int, n_sensors: int = 21) -> str:
    """Returns a space-separated row with dummy sensor values."""
    op_settings = "0.001 -0.002 100.0"
    sensors = " ".join([f"{1.0 + 0.01 * i:.4f}" for i in range(n_sensors)])
    return f"{unit} {cycle} {op_settings} {sensors}"


def _write_train_file(path: Path, engine_specs: list) -> None:
    """
    Writes a minimal train file.

    Parameters
    ----------
    engine_specs : list of (engine_id, n_cycles)
    """
    rows = []
    for eng_id, n_cycles in engine_specs:
        for cyc in range(1, n_cycles + 1):
            rows.append(_make_cmapss_row(eng_id, cyc))
    path.write_text("\n".join(rows))


def _write_test_file(path: Path, engine_specs: list) -> None:
    """
    Writes a minimal test file (engines don't run to failure).
    """
    rows = []
    for eng_id, n_cycles in engine_specs:
        for cyc in range(1, n_cycles + 1):
            rows.append(_make_cmapss_row(eng_id, cyc))
    path.write_text("\n".join(rows))


def _write_rul_file(path: Path, rul_values: list) -> None:
    """Writes one integer per line."""
    path.write_text("\n".join(str(v) for v in rul_values))


@pytest.fixture
def synthetic_dataset(tmp_path):
    """
    Creates a minimal synthetic FD001 dataset in a temp directory.
    3 training engines (cycles 10, 15, 20), 3 test engines (cycles 5 each).
    RUL ground truth: [50, 80, 30].
    """
    # Training: 3 engines
    train_specs = [(1, 10), (2, 15), (3, 20)]
    test_specs = [(1, 5), (2, 5), (3, 5)]
    rul_values = [50, 80, 30]

    _write_train_file(tmp_path / "train_FD001.txt", train_specs)
    _write_test_file(tmp_path / "test_FD001.txt", test_specs)
    _write_rul_file(tmp_path / "RUL_FD001.txt", rul_values)

    return tmp_path, train_specs, test_specs, rul_values


# ---------------------------------------------------------------------------
# Test: Missing directory / files
# ---------------------------------------------------------------------------

class TestFileValidation:

    def test_missing_dataset_dir_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            CMAPSSLoader(dataset_dir="/nonexistent/path/to/cmapss")

    def test_missing_train_file_raises(self, tmp_path):
        # Create only the test and RUL files
        (tmp_path / "test_FD001.txt").write_text(_make_cmapss_row(1, 1) + "\n")
        (tmp_path / "RUL_FD001.txt").write_text("100\n")
        with pytest.raises(FileNotFoundError, match="train_FD001"):
            CMAPSSLoader(dataset_dir=str(tmp_path))

    def test_missing_rul_file_raises(self, tmp_path):
        _write_train_file(tmp_path / "train_FD001.txt", [(1, 5)])
        _write_test_file(tmp_path / "test_FD001.txt", [(1, 3)])
        with pytest.raises(FileNotFoundError, match="RUL_FD001"):
            CMAPSSLoader(dataset_dir=str(tmp_path))

    def test_wrong_column_count_raises(self, tmp_path):
        # Write a file with only 10 columns
        bad_row = " ".join(["1.0"] * 10)
        (tmp_path / "train_FD001.txt").write_text(bad_row + "\n")
        _write_test_file(tmp_path / "test_FD001.txt", [(1, 3)])
        _write_rul_file(tmp_path / "RUL_FD001.txt", [50])
        loader = CMAPSSLoader(dataset_dir=str(tmp_path))
        with pytest.raises(ValueError, match="26 columns"):
            _ = loader.train_df


# ---------------------------------------------------------------------------
# Test: Column names & types
# ---------------------------------------------------------------------------

class TestColumnNames:

    def test_train_column_names(self, synthetic_dataset):
        tmp_path, train_specs, test_specs, rul_values = synthetic_dataset
        loader = CMAPSSLoader(dataset_dir=str(tmp_path), rul_cap=None)
        df = loader.train_df
        # Should have all base columns + 'rul'
        for col in _BASE_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"
        assert "rul" in df.columns

    def test_test_column_names_no_rul(self, synthetic_dataset):
        tmp_path, *_ = synthetic_dataset
        loader = CMAPSSLoader(dataset_dir=str(tmp_path))
        df = loader.test_df
        assert "rul" not in df.columns, "Test DataFrame should NOT have 'rul' column"
        for col in _BASE_COLUMNS:
            assert col in df.columns

    def test_unit_number_dtype_int(self, synthetic_dataset):
        tmp_path, *_ = synthetic_dataset
        loader = CMAPSSLoader(dataset_dir=str(tmp_path))
        assert loader.train_df["unit_number"].dtype in (np.int32, np.int64, int)

    def test_cycle_dtype_int(self, synthetic_dataset):
        tmp_path, *_ = synthetic_dataset
        loader = CMAPSSLoader(dataset_dir=str(tmp_path))
        assert loader.train_df["cycle"].dtype in (np.int32, np.int64, int)


# ---------------------------------------------------------------------------
# Test: RUL Label Computation
# ---------------------------------------------------------------------------

class TestRULLabels:

    def test_rul_at_last_cycle_is_zero(self, synthetic_dataset):
        """For each training engine, RUL at its last cycle should be 0."""
        tmp_path, train_specs, *_ = synthetic_dataset
        loader = CMAPSSLoader(dataset_dir=str(tmp_path), rul_cap=None)
        df = loader.train_df
        for eng_id, n_cycles in train_specs:
            eng_df = df[df["unit_number"] == eng_id]
            last_row = eng_df[eng_df["cycle"] == n_cycles]
            assert len(last_row) == 1
            assert last_row["rul"].iloc[0] == 0, (
                f"Engine {eng_id}: RUL at last cycle should be 0, "
                f"got {last_row['rul'].iloc[0]}"
            )

    def test_rul_at_first_cycle_equals_max_minus_one(self, synthetic_dataset):
        """For each training engine, RUL at cycle 1 = (max_cycle - 1)."""
        tmp_path, train_specs, *_ = synthetic_dataset
        loader = CMAPSSLoader(dataset_dir=str(tmp_path), rul_cap=None)
        df = loader.train_df
        for eng_id, n_cycles in train_specs:
            eng_df = df[df["unit_number"] == eng_id]
            first_row = eng_df[eng_df["cycle"] == 1]
            expected_rul = n_cycles - 1
            assert first_row["rul"].iloc[0] == expected_rul, (
                f"Engine {eng_id}: RUL at cycle 1 should be {expected_rul}, "
                f"got {first_row['rul'].iloc[0]}"
            )

    def test_rul_monotonically_decreasing_per_engine(self, synthetic_dataset):
        """RUL must be non-increasing along each engine's trajectory."""
        tmp_path, train_specs, *_ = synthetic_dataset
        loader = CMAPSSLoader(dataset_dir=str(tmp_path), rul_cap=None)
        df = loader.train_df
        for eng_id, _ in train_specs:
            eng_df = df[df["unit_number"] == eng_id].sort_values("cycle")
            rul_vals = eng_df["rul"].values
            assert all(
                rul_vals[i] >= rul_vals[i + 1] for i in range(len(rul_vals) - 1)
            ), f"Engine {eng_id}: RUL should be non-increasing."

    def test_no_negative_rul(self, synthetic_dataset):
        """RUL labels must all be >= 0."""
        tmp_path, *_ = synthetic_dataset
        loader = CMAPSSLoader(dataset_dir=str(tmp_path), rul_cap=None)
        assert (loader.train_df["rul"] >= 0).all(), "Negative RUL labels found!"

    def test_rul_cap_applied(self, synthetic_dataset):
        """RUL values should not exceed rul_cap when cap is set."""
        tmp_path, *_ = synthetic_dataset
        cap = 5
        loader = CMAPSSLoader(dataset_dir=str(tmp_path), rul_cap=cap)
        assert (loader.train_df["rul"] <= cap).all(), f"RUL exceeds cap={cap}"

    def test_rul_cap_none_disables_cap(self, synthetic_dataset):
        """With rul_cap=None, RUL values may exceed 125."""
        tmp_path, train_specs, *_ = synthetic_dataset
        loader = CMAPSSLoader(dataset_dir=str(tmp_path), rul_cap=None)
        max_n_cycles = max(n for _, n in train_specs)
        # The maximum possible RUL is max_cycle - 1
        max_possible_rul = max_n_cycles - 1
        assert loader.train_df["rul"].max() == max_possible_rul

    def test_no_cross_engine_contamination(self, synthetic_dataset):
        """RUL for engine A must not be influenced by engine B's max cycle."""
        tmp_path, train_specs, *_ = synthetic_dataset
        loader = CMAPSSLoader(dataset_dir=str(tmp_path), rul_cap=None)
        df = loader.train_df
        for eng_id, n_cycles in train_specs:
            eng_df = df[df["unit_number"] == eng_id]
            max_rul = eng_df["rul"].max()
            expected_max = n_cycles - 1
            assert max_rul == expected_max, (
                f"Engine {eng_id}: max RUL={max_rul} but expected {expected_max}. "
                "Cross-engine contamination suspected."
            )


# ---------------------------------------------------------------------------
# Test: Ground-Truth RUL Loading
# ---------------------------------------------------------------------------

class TestGroundTruthRUL:

    def test_rul_ground_truth_shape(self, synthetic_dataset):
        tmp_path, _, _, rul_values = synthetic_dataset
        loader = CMAPSSLoader(dataset_dir=str(tmp_path))
        rul_gt = loader.rul_ground_truth
        assert len(rul_gt) == len(rul_values)

    def test_rul_ground_truth_values(self, synthetic_dataset):
        tmp_path, _, _, rul_values = synthetic_dataset
        loader = CMAPSSLoader(dataset_dir=str(tmp_path))
        np.testing.assert_array_equal(loader.rul_ground_truth, np.array(rul_values))

    def test_rul_ground_truth_dtype_int(self, synthetic_dataset):
        tmp_path, *_ = synthetic_dataset
        loader = CMAPSSLoader(dataset_dir=str(tmp_path))
        assert loader.rul_ground_truth.dtype == np.int32

    def test_empty_rul_file_raises(self, tmp_path):
        _write_train_file(tmp_path / "train_FD001.txt", [(1, 5)])
        _write_test_file(tmp_path / "test_FD001.txt", [(1, 3)])
        (tmp_path / "RUL_FD001.txt").write_text("")  # empty
        loader = CMAPSSLoader(dataset_dir=str(tmp_path))
        with pytest.raises(ValueError, match="empty"):
            _ = loader.rul_ground_truth


# ---------------------------------------------------------------------------
# Test: Summary & API
# ---------------------------------------------------------------------------

class TestLoaderAPI:

    def test_load_all_returns_three_items(self, synthetic_dataset):
        tmp_path, *_ = synthetic_dataset
        loader = CMAPSSLoader(dataset_dir=str(tmp_path))
        result = loader.load_all()
        assert len(result) == 3

    def test_get_engine_trajectory_returns_sorted_cycles(self, synthetic_dataset):
        tmp_path, train_specs, *_ = synthetic_dataset
        loader = CMAPSSLoader(dataset_dir=str(tmp_path))
        eng_df = loader.get_engine_trajectory(1, split="train")
        cycles = eng_df["cycle"].values
        assert all(cycles[i] < cycles[i + 1] for i in range(len(cycles) - 1))

    def test_get_engine_trajectory_invalid_id_raises(self, synthetic_dataset):
        tmp_path, *_ = synthetic_dataset
        loader = CMAPSSLoader(dataset_dir=str(tmp_path))
        with pytest.raises(ValueError, match="not found"):
            loader.get_engine_trajectory(999, split="train")

    def test_summary_keys_present(self, synthetic_dataset):
        tmp_path, *_ = synthetic_dataset
        loader = CMAPSSLoader(dataset_dir=str(tmp_path))
        summ = loader.summary()
        for key in ["train_engines", "train_total_cycles", "test_engines", "rul_gt_mean"]:
            assert key in summ, f"Missing summary key: {key}"

    def test_load_fd001_function(self, synthetic_dataset):
        tmp_path, *_ = synthetic_dataset
        train_df, test_df, rul_gt = load_fd001(dataset_dir=str(tmp_path))
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert isinstance(rul_gt, np.ndarray)
        assert "rul" in train_df.columns
        assert "rul" not in test_df.columns
