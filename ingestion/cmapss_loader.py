"""
AeroVigil - NASA C-MAPSS FD001 Dataset Loader
ingestion/cmapss_loader.py

Loads, validates, and labels the NASA C-MAPSS FD001 train/test datasets.
Provides a clean adapter interface so downstream AeroVigil PHM modules
consume a standardized representation independent of raw dataset layout.

Domain Disclaimer:
    C-MAPSS FD001 describes turbofan HPC degradation, NOT aero-piston engine
    physics. This module is used solely to validate generic PHM capabilities
    (degradation modelling, RUL prediction, anomaly detection) of the
    AeroVigil prototype. It does NOT calibrate or replace the piston-engine
    Digital Twin or UAV mission-reliability layer.

Dataset Layout (26 space-separated columns, no header):
    Col 0:  unit_number   (engine ID, 1-indexed)
    Col 1:  cycle         (operational cycle number)
    Col 2:  op_setting_1  (operational setting)
    Col 3:  op_setting_2  (operational setting)
    Col 4:  op_setting_3  (operational setting)
    Col 5-25: s1–s21      (21 sensor measurements)

RUL_FD001.txt:
    One integer RUL per line, corresponding to test engine IDs 1–100.
    Each value = number of cycles remaining AFTER the last observed test cycle.
    These values are NEVER used during model training; only for evaluation.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column Schema
# ---------------------------------------------------------------------------
# Canonical column names assigned after loading (matches FD001 readme)
_BASE_COLUMNS: List[str] = [
    "unit_number",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"s{i}" for i in range(1, 22)]  # s1 … s21

_EXPECTED_N_COLS: int = 26  # 1 + 1 + 3 + 21 = 26

# Default dataset location relative to AeroVigil repo root
_DEFAULT_DATASET_DIR: str = "official_dataset_NASA"


# ---------------------------------------------------------------------------
# CMAPSSLoader
# ---------------------------------------------------------------------------

class CMAPSSLoader:
    """
    NASA C-MAPSS Dataset Loader and Validator.

    Responsibilities:
        1. Load raw train/test text files and assign canonical column names.
        2. Validate structure (column count, engine IDs, cycle ordering, NaN).
        3. Compute cycle-level RUL labels for training trajectories.
        4. Load and expose ground-truth RUL values from RUL_FD00x.txt.
        5. Provide a single clean DataFrame to downstream preprocessing.

    Design Principle:
        All FD001-specific assumptions are isolated here.
        Downstream modules consume standard DataFrames with known column names.

    Parameters
    ----------
    dataset_dir : str or Path
        Path to the directory containing the C-MAPSS .txt files.
        Defaults to ``official_dataset_NASA/`` under the repo root.
    fd_id : str
        Dataset identifier. Only "FD001" is fully validated; FD002–FD004
        may be loaded but have different operating condition counts.
    rul_cap : int or None
        Maximum RUL value to assign for training labels.
        Common PHM practice caps RUL at 125 to reduce early-trajectory noise.
        Set to None to disable capping.
    """

    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        fd_id: str = "FD001",
        rul_cap: Optional[int] = 125,
    ):
        self.fd_id = fd_id.upper()
        self.rul_cap = rul_cap

        # Resolve dataset directory
        if dataset_dir is None:
            # Walk up from this file to find repo root, then locate dataset dir
            repo_root = Path(__file__).resolve().parent.parent
            self.dataset_dir = repo_root / _DEFAULT_DATASET_DIR
        else:
            self.dataset_dir = Path(dataset_dir)

        self._validate_dataset_dir()

        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._rul_ground_truth: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    @property
    def train_df(self) -> pd.DataFrame:
        """Training trajectories with RUL labels. Loaded lazily."""
        if self._train_df is None:
            self._train_df = self._load_train()
        return self._train_df

    @property
    def test_df(self) -> pd.DataFrame:
        """Test trajectories (no RUL labels). Loaded lazily."""
        if self._test_df is None:
            self._test_df = self._load_test()
        return self._test_df

    @property
    def rul_ground_truth(self) -> np.ndarray:
        """
        Array of shape (n_test_engines,) with true RUL values.
        Index i corresponds to test engine (i+1).
        These values are ONLY for evaluation; never for model training.
        """
        if self._rul_ground_truth is None:
            self._rul_ground_truth = self._load_rul_ground_truth()
        return self._rul_ground_truth

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Convenience method: loads and returns train_df, test_df, rul_ground_truth.

        Returns
        -------
        train_df : pd.DataFrame
            Training trajectories with ``rul`` column (cycle-level labels).
        test_df : pd.DataFrame
            Test trajectories (no ``rul`` column).
        rul_gt : np.ndarray
            Ground-truth RUL values for the 100 test engines.
        """
        return self.train_df, self.test_df, self.rul_ground_truth

    def get_engine_trajectory(
        self,
        engine_id: int,
        split: str = "train",
    ) -> pd.DataFrame:
        """
        Returns all rows for a specific engine from train or test split.

        Parameters
        ----------
        engine_id : int
            Engine unit number (1-indexed, 1–100 for FD001).
        split : str
            "train" or "test".

        Returns
        -------
        pd.DataFrame
            Rows for the specified engine, sorted by cycle.
        """
        df = self.train_df if split == "train" else self.test_df
        eng_df = df[df["unit_number"] == engine_id].copy()
        if eng_df.empty:
            raise ValueError(
                f"Engine {engine_id} not found in {split} split of {self.fd_id}."
            )
        return eng_df.sort_values("cycle").reset_index(drop=True)

    def summary(self) -> Dict:
        """Returns a summary dictionary of dataset statistics."""
        tr = self.train_df
        te = self.test_df
        return {
            "fd_id": self.fd_id,
            "dataset_dir": str(self.dataset_dir),
            "rul_cap": self.rul_cap,
            "train_engines": int(tr["unit_number"].nunique()),
            "train_total_cycles": len(tr),
            "train_min_trajectory_len": int(tr.groupby("unit_number")["cycle"].max().min()),
            "train_max_trajectory_len": int(tr.groupby("unit_number")["cycle"].max().max()),
            "train_mean_trajectory_len": float(tr.groupby("unit_number")["cycle"].max().mean()),
            "test_engines": int(te["unit_number"].nunique()),
            "test_total_cycles": len(te),
            "n_sensors": 21,
            "n_op_settings": 3,
            "rul_gt_min": int(self.rul_ground_truth.min()),
            "rul_gt_max": int(self.rul_ground_truth.max()),
            "rul_gt_mean": float(self.rul_ground_truth.mean()),
        }

    # ------------------------------------------------------------------
    # Private: Loading
    # ------------------------------------------------------------------

    def _validate_dataset_dir(self) -> None:
        """Checks that the dataset directory and expected files exist."""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"C-MAPSS dataset directory not found: {self.dataset_dir}\n"
                "Please ensure the dataset files are placed under "
                "'official_dataset_NASA/' in the AeroVigil repository root."
            )

        train_file = self.dataset_dir / f"train_{self.fd_id}.txt"
        test_file = self.dataset_dir / f"test_{self.fd_id}.txt"
        rul_file = self.dataset_dir / f"RUL_{self.fd_id}.txt"

        missing = [f for f in [train_file, test_file, rul_file] if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing C-MAPSS {self.fd_id} files:\n"
                + "\n".join(f"  {p}" for p in missing)
            )

    def _load_raw(self, split: str) -> pd.DataFrame:
        """
        Loads a raw C-MAPSS text file, strips trailing NaN columns,
        and assigns canonical column names.

        Parameters
        ----------
        split : str
            "train" or "test".

        Returns
        -------
        pd.DataFrame with columns as per _BASE_COLUMNS.
        """
        filepath = self.dataset_dir / f"{split}_{self.fd_id}.txt"
        logger.info("Loading %s %s from %s", self.fd_id, split, filepath)

        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            engine="python",
        )

        # Drop all-NaN trailing columns (C-MAPSS files sometimes have a
        # trailing space that produces an extra empty column)
        df = df.dropna(axis=1, how="all")

        # Validate column count
        n_cols = df.shape[1]
        if n_cols != _EXPECTED_N_COLS:
            raise ValueError(
                f"Expected {_EXPECTED_N_COLS} columns in {filepath.name}, "
                f"got {n_cols}. Check file integrity."
            )

        df.columns = _BASE_COLUMNS

        # Validate data types
        for col in _BASE_COLUMNS:
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError as exc:
                    raise ValueError(
                        f"Column '{col}' in {filepath.name} contains "
                        f"non-numeric values: {exc}"
                    ) from exc

        # Validate no NaN in sensor/cycle columns
        nan_counts = df.isnull().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        if not cols_with_nan.empty:
            logger.warning(
                "NaN values found in %s %s:\n%s\nRows with NaN will be dropped.",
                self.fd_id, split, cols_with_nan
            )
            n_before = len(df)
            df = df.dropna().reset_index(drop=True)
            logger.warning("Dropped %d rows with NaN values.", n_before - len(df))

        # Validate engine IDs are positive integers
        if (df["unit_number"] <= 0).any():
            raise ValueError(
                f"Non-positive engine IDs found in {filepath.name}. "
                "Dataset may be malformed."
            )

        # Validate cycle numbers are positive
        if (df["cycle"] <= 0).any():
            raise ValueError(
                f"Non-positive cycle values found in {filepath.name}."
            )

        # Cast IDs to int
        df["unit_number"] = df["unit_number"].astype(int)
        df["cycle"] = df["cycle"].astype(int)

        logger.info(
            "Loaded %s %s: %d rows, %d engines",
            self.fd_id, split, len(df), df["unit_number"].nunique()
        )
        return df

    def _load_train(self) -> pd.DataFrame:
        """Loads training data and computes cycle-level RUL labels."""
        df = self._load_raw("train")

        # Compute RUL for each engine:
        # RUL = max_cycle_for_engine - current_cycle
        # This is the standard PHM label for run-to-failure training data.
        max_cycles = df.groupby("unit_number")["cycle"].max().rename("max_cycle")
        df = df.join(max_cycles, on="unit_number")
        df["rul"] = df["max_cycle"] - df["cycle"]
        df.drop(columns=["max_cycle"], inplace=True)

        # Apply RUL cap if configured (reduces early-trajectory noise,
        # per standard PHM/NASA PHMAP convention)
        if self.rul_cap is not None:
            df["rul"] = df["rul"].clip(upper=self.rul_cap)
            logger.info(
                "RUL labels capped at %d cycles (piecewise-linear clipping).",
                self.rul_cap,
            )

        return df

    def _load_test(self) -> pd.DataFrame:
        """Loads test data (no RUL labels - those come from RUL_FD001.txt)."""
        return self._load_raw("test")

    def _load_rul_ground_truth(self) -> np.ndarray:
        """
        Loads RUL_FD001.txt as a numpy array.

        Returns
        -------
        np.ndarray of shape (n_test_engines,)
            Index 0 → engine 1, index 99 → engine 100.
        """
        rul_file = self.dataset_dir / f"RUL_{self.fd_id}.txt"
        logger.info("Loading ground-truth RUL from %s", rul_file)

        values = []
        with open(rul_file, "r") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        values.append(int(line))
                    except ValueError as exc:
                        raise ValueError(
                            f"Non-integer value '{line}' in {rul_file.name}: {exc}"
                        ) from exc

        rul_array = np.array(values, dtype=np.int32)

        if len(rul_array) == 0:
            raise ValueError(f"{rul_file.name} is empty.")

        logger.info(
            "Loaded %d ground-truth RUL values. Min=%d, Max=%d, Mean=%.1f",
            len(rul_array), rul_array.min(), rul_array.max(), rul_array.mean()
        )
        return rul_array


# ---------------------------------------------------------------------------
# Convenience Functional API
# ---------------------------------------------------------------------------

def load_fd001(
    dataset_dir: Optional[str] = None,
    rul_cap: Optional[int] = 125,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Convenience function: loads FD001 train/test and ground-truth RUL.

    Parameters
    ----------
    dataset_dir : str or None
        Path to the directory containing C-MAPSS .txt files.
        Defaults to 'official_dataset_NASA/' under the repo root.
    rul_cap : int or None
        Maximum RUL label for training data. Default 125 cycles.

    Returns
    -------
    train_df : pd.DataFrame
        Training trajectories with ``rul`` column.
    test_df : pd.DataFrame
        Test trajectories.
    rul_gt : np.ndarray
        100 ground-truth RUL values for evaluation (never for training).

    Example
    -------
    >>> from ingestion.cmapss_loader import load_fd001
    >>> train_df, test_df, rul_gt = load_fd001()
    >>> print(train_df.shape)      # (20631, 27)  — 26 cols + rul
    >>> print(rul_gt[:5])           # [112, 98, 69, 82, 91]
    """
    loader = CMAPSSLoader(dataset_dir=dataset_dir, fd_id="FD001", rul_cap=rul_cap)
    return loader.load_all()
