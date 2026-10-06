"""
AeroVigil - NASA C-MAPSS FD001 Preprocessing Pipeline
ingestion/cmapss_preprocessor.py

Performs exploratory analysis, sensor selection, normalization, and
healthy-baseline extraction for the FD001 dataset.

Preprocessing Decisions (all documented, none silent):
    1. Identify constant / near-constant sensors → drop (documented).
    2. Select informative sensors based on per-engine trajectory variance.
    3. Fit StandardScaler on training data only (no test leakage).
    4. Extract "healthy baseline" from first N_HEALTHY_CYCLES cycles of
       each training engine (before degradation sets in).
    5. Compute per-sensor healthy-baseline mean and std for the
       residual adapter.

Design Constraint:
    Scalers and baseline statistics are fit ONLY on training engines.
    Transformation is applied to test data using training statistics.
    No information from test-engine RUL labels is used here.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All 21 sensor column names in FD001
ALL_SENSOR_COLS: List[str] = [f"s{i}" for i in range(1, 22)]

# Operating setting columns
OP_SETTING_COLS: List[str] = ["op_setting_1", "op_setting_2", "op_setting_3"]

# Threshold: sensors with std < this across ALL training data are dropped
CONSTANT_SENSOR_STD_THRESHOLD: float = 0.001

# Number of early cycles per engine to define the "healthy baseline"
# (before significant degradation develops)
N_HEALTHY_CYCLES: int = 30


class CMAPSSPreprocessor:
    """
    FD001 Preprocessing and Exploratory Analysis Pipeline.

    Follows strict training-first discipline:
    - fit() operates on training data only.
    - transform() applies fitted parameters to any split.
    - No test data information reaches fit() at any point.

    Attributes
    ----------
    selected_sensors : List[str]
        Sensor columns retained after constant-sensor removal.
    dropped_sensors : List[str]
        Sensor columns removed (near-constant across all training data).
    sensor_means : pd.Series
        Per-sensor mean from training data (for normalization).
    sensor_stds : pd.Series
        Per-sensor std from training data (for normalization).
    healthy_baseline_mean : pd.Series
        Mean sensor values from healthy early cycles (training engines only).
    healthy_baseline_std : pd.Series
        Std sensor values from healthy early cycles (training engines only).
    is_fitted : bool
        True after fit() has been called.
    """

    def __init__(self):
        self.selected_sensors: List[str] = []
        self.dropped_sensors: List[str] = []
        self.sensor_means: Optional[pd.Series] = None
        self.sensor_stds: Optional[pd.Series] = None
        self.healthy_baseline_mean: Optional[pd.Series] = None
        self.healthy_baseline_std: Optional[pd.Series] = None
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "CMAPSSPreprocessor":
        """
        Fits the preprocessing pipeline on training data.

        Steps
        -----
        1. Identify and remove constant sensors.
        2. Compute training-set sensor statistics for z-score normalization.
        3. Extract healthy-baseline statistics from early training cycles.

        Parameters
        ----------
        train_df : pd.DataFrame
            Raw training DataFrame from CMAPSSLoader (with 'rul' column).

        Returns
        -------
        self (for chaining)
        """
        logger.info("Fitting CMAPSSPreprocessor on %d training rows, %d engines.",
                    len(train_df), train_df["unit_number"].nunique())

        # Step 1: Identify constant / near-constant sensors
        sensor_stds_all = train_df[ALL_SENSOR_COLS].std()
        self.dropped_sensors = list(
            sensor_stds_all[sensor_stds_all < CONSTANT_SENSOR_STD_THRESHOLD].index
        )
        self.selected_sensors = [
            s for s in ALL_SENSOR_COLS if s not in self.dropped_sensors
        ]

        logger.info(
            "Dropped %d constant/near-constant sensors: %s",
            len(self.dropped_sensors),
            self.dropped_sensors,
        )
        logger.info(
            "Retained %d informative sensors: %s",
            len(self.selected_sensors),
            self.selected_sensors,
        )

        # Step 2: Compute global training statistics for z-score normalization
        self.sensor_means = train_df[self.selected_sensors].mean()
        self.sensor_stds = train_df[self.selected_sensors].std().replace(0, 1.0)

        logger.info("Computed training-set normalization statistics.")

        # Step 3: Extract healthy baseline from early cycles of each engine
        # "Healthy" = first N_HEALTHY_CYCLES cycles per engine (before degradation)
        healthy_mask = train_df.groupby("unit_number")["cycle"].transform(
            lambda c: c <= N_HEALTHY_CYCLES
        )
        healthy_data = train_df.loc[healthy_mask.astype(bool), self.selected_sensors]

        if len(healthy_data) == 0:
            raise ValueError(
                f"No healthy baseline data found. N_HEALTHY_CYCLES={N_HEALTHY_CYCLES} "
                "may be larger than the shortest engine trajectory."
            )

        self.healthy_baseline_mean = healthy_data.mean()
        self.healthy_baseline_std = healthy_data.std().replace(0, 1.0)

        n_healthy_engines = train_df.loc[healthy_mask.astype(bool), "unit_number"].nunique()
        logger.info(
            "Healthy baseline extracted from first %d cycles of %d engines "
            "(%d total healthy rows).",
            N_HEALTHY_CYCLES, n_healthy_engines, len(healthy_data)
        )

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies fitted preprocessing to a DataFrame (train or test).

        Adds columns:
            - ``{sensor}_norm``:     z-score normalized sensor value
            - ``{sensor}_residual``: deviation from healthy-baseline mean
            - ``{sensor}_z``:        residual normalized by healthy-baseline std
                                     (analogous to AeroVigil normalized_residual)

        Parameters
        ----------
        df : pd.DataFrame
            Raw C-MAPSS DataFrame (from CMAPSSLoader).

        Returns
        -------
        pd.DataFrame
            Copy of input DataFrame with added normalized and residual columns.
            Only selected (non-constant) sensors are processed.
            Dropped sensors are preserved in the output but not transformed.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "CMAPSSPreprocessor.fit() must be called before transform()."
            )

        result = df.copy()

        for col in self.selected_sensors:
            # Z-score normalization (training statistics)
            result[f"{col}_norm"] = (
                (result[col] - self.sensor_means[col]) / self.sensor_stds[col]
            )
            # Deviation from healthy baseline
            result[f"{col}_residual"] = result[col] - self.healthy_baseline_mean[col]
            # Healthy-baseline normalized residual (equivalent to AeroVigil normalized_residual)
            result[f"{col}_z"] = (
                result[f"{col}_residual"] / self.healthy_baseline_std[col]
            )

        return result

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fits on train_df and transforms it in one call."""
        return self.fit(train_df).transform(train_df)

    def get_eda_summary(self, train_df: pd.DataFrame) -> Dict:
        """
        Returns an exploratory data analysis summary of the training dataset.
        Does NOT require fit() to have been called first.

        Returns
        -------
        dict with keys:
            n_engines, total_cycles, trajectory_lengths (stats),
            sensor_stds (per sensor), constant_sensors, informative_sensors,
            op_setting_stats, rul_label_stats
        """
        traj_lens = train_df.groupby("unit_number")["cycle"].max()
        sensor_stds_all = train_df[ALL_SENSOR_COLS].std()
        const_sensors = list(
            sensor_stds_all[sensor_stds_all < CONSTANT_SENSOR_STD_THRESHOLD].index
        )

        summary = {
            "n_engines": int(train_df["unit_number"].nunique()),
            "total_cycles": len(train_df),
            "trajectory_lengths": {
                "min": int(traj_lens.min()),
                "max": int(traj_lens.max()),
                "mean": float(traj_lens.mean()),
                "std": float(traj_lens.std()),
            },
            "sensor_stds": sensor_stds_all.to_dict(),
            "constant_sensors": const_sensors,
            "informative_sensors": [s for s in ALL_SENSOR_COLS if s not in const_sensors],
            "op_setting_stats": {
                col: {
                    "unique": int(train_df[col].nunique()),
                    "std": float(train_df[col].std()),
                }
                for col in OP_SETTING_COLS
            },
        }

        if "rul" in train_df.columns:
            summary["rul_label_stats"] = {
                "min": float(train_df["rul"].min()),
                "max": float(train_df["rul"].max()),
                "mean": float(train_df["rul"].mean()),
                "std": float(train_df["rul"].std()),
            }

        return summary

    def build_windowed_features(
        self,
        df: pd.DataFrame,
        window_size: int = 15,
        step: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds sliding-window feature arrays for RUL model training/prediction.

        For each engine, slides a window of size ``window_size`` over its
        trajectory and computes mean + std of each selected sensor across
        the window. This creates 2 * len(selected_sensors) features per sample.

        Data Leakage Prevention:
            - Windows are computed PER ENGINE sequentially.
            - The last row of each window provides its label (RUL at that cycle).
            - No cross-engine contamination possible.

        Parameters
        ----------
        df : pd.DataFrame
            Transformed DataFrame (output of transform()). Must contain
            ``{sensor}_norm`` columns and a ``rul`` column.
        window_size : int
            Number of consecutive cycles in each window.
        step : int
            Stride between windows (1 = dense, no skipping).

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)  — RUL labels
        unit_ids : np.ndarray, shape (n_samples,) — engine ID for each sample
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before build_windowed_features().")

        norm_cols = [f"{s}_norm" for s in self.selected_sensors]
        missing = [c for c in norm_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Normalized columns not found in DataFrame: {missing}. "
                "Run transform() first."
            )

        has_rul = "rul" in df.columns

        X_list, y_list, uid_list = [], [], []

        for engine_id, eng_df in df.groupby("unit_number"):
            eng_df = eng_df.sort_values("cycle").reset_index(drop=True)
            n = len(eng_df)

            if n < window_size:
                logger.debug(
                    "Engine %d has only %d cycles (< window_size=%d). Skipping.",
                    engine_id, n, window_size,
                )
                continue

            sensor_vals = eng_df[norm_cols].values  # shape (n, n_sensors)
            rul_vals = eng_df["rul"].values if has_rul else np.zeros(n)

            for start in range(0, n - window_size + 1, step):
                end = start + window_size
                window = sensor_vals[start:end]  # (window_size, n_sensors)

                # Features: mean and std over window for each sensor
                feats = np.concatenate([window.mean(axis=0), window.std(axis=0)])
                label = rul_vals[end - 1]

                X_list.append(feats)
                y_list.append(label)
                uid_list.append(engine_id)

        if not X_list:
            raise ValueError(
                "No windowed samples could be constructed. "
                f"Check that engines have >= {window_size} cycles."
            )

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        unit_ids = np.array(uid_list, dtype=np.int32)

        logger.info(
            "Built %d windowed samples from %d engines (window_size=%d).",
            len(X), len(np.unique(unit_ids)), window_size,
        )
        return X, y, unit_ids

    def build_last_window_features(
        self,
        df: pd.DataFrame,
        window_size: int = 15,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds a single feature vector from the LAST window of each engine.
        Used for test-set prediction (one prediction per engine).

        Parameters
        ----------
        df : pd.DataFrame
            Transformed test DataFrame (output of transform()).
        window_size : int
            Must match the window size used during training.

        Returns
        -------
        X : np.ndarray, shape (n_engines, n_features)
        unit_ids : np.ndarray, shape (n_engines,)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before build_last_window_features().")

        norm_cols = [f"{s}_norm" for s in self.selected_sensors]
        missing = [c for c in norm_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Normalized columns not found. Run transform() first: {missing}"
            )

        X_list, uid_list = [], []

        for engine_id, eng_df in df.groupby("unit_number"):
            eng_df = eng_df.sort_values("cycle").reset_index(drop=True)
            n = len(eng_df)
            sensor_vals = eng_df[norm_cols].values

            if n >= window_size:
                window = sensor_vals[-window_size:]
            else:
                # Pad with first available row if engine is shorter than window
                pad = np.tile(sensor_vals[0], (window_size - n, 1))
                window = np.vstack([pad, sensor_vals])
                logger.debug(
                    "Engine %d: padded %d cycles to reach window_size=%d.",
                    engine_id, n, window_size
                )

            feats = np.concatenate([window.mean(axis=0), window.std(axis=0)])
            X_list.append(feats)
            uid_list.append(engine_id)

        X = np.array(X_list, dtype=np.float32)
        unit_ids = np.array(uid_list, dtype=np.int32)

        logger.info(
            "Built last-window features for %d test engines.", len(X)
        )
        return X, unit_ids
