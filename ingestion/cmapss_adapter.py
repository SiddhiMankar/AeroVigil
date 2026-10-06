"""
AeroVigil - C-MAPSS → AeroVigil Residual Bridge
ingestion/cmapss_adapter.py

Translates FD001 preprocessed data into the AeroVigil residual representation
so that the existing EngineHealthIndex module can consume it without modification.

Architecture Note:
    The AeroVigil Digital Twin (digital_twin.py) uses a physics-based surrogate
    to estimate "expected" state from throttle + ambient temperature. FD001 does
    NOT provide these physical inputs, so the physics model cannot be applied.

    Instead, this adapter implements a DATA-DRIVEN baseline approach:
        expected_state = healthy_baseline_mean  (from early training cycles)
        residual       = observed - expected_state
        norm_residual  = residual / healthy_baseline_std

    This is conceptually identical to the AeroVigil residual pipeline, but
    the "expected" comes from a data-derived fleet healthy mean, not a
    physics model. This distinction is clearly documented everywhere.

    The adapter maps FD001 sensors to a synthetic AeroVigil-compatible
    signal representation, then invokes the existing EngineHealthIndex
    to produce a 0-100 health index.

    IMPORTANT: This health index is labeled as "Model-Derived Health Index
    (Data-Driven)" to distinguish it from a physically measured health value.

Sensor Mapping Strategy:
    FD001 has HPC degradation (turbofan). AeroVigil has piston-engine sensors.
    We do NOT attempt to claim physical equivalence. Instead we use the
    degradation-correlated FD001 sensors directly to compute a
    "PHM validation health index" through the same pipeline math.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.health_index import EngineHealthIndex
import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anomaly scoring parameters (adapted for FD001 cycle-domain data)
# We reuse the same threshold approach as AeroVigil's ResidualAnalyzer
# but applied to FD001 normalized sensor residuals.
# ---------------------------------------------------------------------------
WARNING_THRESHOLD_Z: float = 2.0    # |z| >= 2.0 → contributing signal
ANOMALY_THRESHOLD_Z: float = 3.0    # |z| >= 3.0 → anomaly flag
CRITICAL_THRESHOLD_Z: float = 5.0   # |z| >= 5.0 → critical flag
PERSISTENCE_WINDOW: int = 5         # cycles for persistence assessment


class CMAPSSResidualAdapter:
    """
    Bridges FD001 preprocessed sensor data into the AeroVigil health pipeline.

    For each engine trajectory (or batch of engines), computes:
        - Per-sensor z-score residuals (deviation from healthy baseline)
        - Global anomaly score (0–100, analogous to AeroVigil)
        - Anomaly level, severity (reusing AeroVigil vocabulary)
        - Model-derived health index (0–100) via EngineHealthIndex
        - Degradation state (monotonically estimated from residual trend)

    Parameters
    ----------
    selected_sensors : list of str
        Sensor columns retained by CMAPSSPreprocessor.
    healthy_baseline_std : pd.Series
        Per-sensor standard deviation of the healthy baseline.
    """

    def __init__(
        self,
        selected_sensors: List[str],
        healthy_baseline_std: pd.Series,
    ):
        self.selected_sensors = selected_sensors
        self.healthy_baseline_std = healthy_baseline_std

    def _compute_anomaly_score(self, z_scores: Dict[str, float]) -> Tuple[float, str, str, float]:
        """
        Computes an anomaly score (0–100) from a set of z-score residuals.

        Uses the same weighted-sum approach as AeroVigil ResidualAnalyzer,
        but with equal sensor weights (no physical domain weighting available
        for FD001 without domain expertise on turbofan HPC).

        Returns
        -------
        anomaly_score : float (0-100)
        anomaly_level : str
        severity : str
        evidence_score : float (0-1)
        """
        weighted_sum = 0.0
        n_sensors = len(self.selected_sensors)
        n_anomalous = 0

        for sig in self.selected_sensors:
            z = abs(z_scores.get(sig, 0.0))
            if z >= WARNING_THRESHOLD_Z:
                excess = z - (WARNING_THRESHOLD_Z - 0.5)
                weighted_sum += excess ** 1.3
            if z >= ANOMALY_THRESHOLD_Z:
                n_anomalous += 1

        if n_sensors > 0:
            raw_score = (weighted_sum / n_sensors) * 22.0
        else:
            raw_score = 0.0

        anomaly_score = float(min(100.0, max(0.0, raw_score)))

        # Anomaly level
        if anomaly_score < 15.0:
            anomaly_level = "NORMAL"
        elif anomaly_score < 35.0:
            anomaly_level = "WARNING"
        elif anomaly_score < 65.0:
            anomaly_level = "ANOMALOUS"
        else:
            anomaly_level = "CRITICAL"

        # Severity (based on fraction of sensors anomalous)
        frac_anomalous = n_anomalous / max(1, n_sensors)
        if frac_anomalous < 0.05:
            severity = "NORMAL"
        elif frac_anomalous < 0.20:
            severity = "LOW"
        elif frac_anomalous < 0.40:
            severity = "MEDIUM"
        elif frac_anomalous < 0.65:
            severity = "HIGH"
        else:
            severity = "CRITICAL"

        # Evidence score: fraction of sensors with z > anomaly threshold
        evidence_score = min(1.0, frac_anomalous * 2.0)

        return anomaly_score, anomaly_level, severity, evidence_score

    def compute_engine_health_trajectory(
        self, engine_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes the health trajectory for a single engine.

        Parameters
        ----------
        engine_df : pd.DataFrame
            Transformed DataFrame for ONE engine (output of CMAPSSPreprocessor.transform()).
            Must be sorted by cycle and contain ``{sensor}_z`` columns.

        Returns
        -------
        pd.DataFrame
            Input DataFrame augmented with:
                anomaly_score, anomaly_level, severity, evidence_score,
                health_index (model-derived), health_state, health_trend,
                health_penalty, fault_type (always "HPC_DEGRADATION" for FD001)
        """
        z_cols = {s: f"{s}_z" for s in self.selected_sensors}
        missing_zcols = [v for v in z_cols.values() if v not in engine_df.columns]
        if missing_zcols:
            raise ValueError(
                f"Missing z-score columns: {missing_zcols}. "
                "Run CMAPSSPreprocessor.transform() first."
            )

        engine_df = engine_df.sort_values("cycle").reset_index(drop=True)
        n = len(engine_df)

        anomaly_scores = np.zeros(n)
        anomaly_levels = []
        severities = []
        evidences = np.zeros(n)
        persistent_flags_list = []

        # Persistence tracking per sensor
        z_history: Dict[str, List[float]] = {s: [] for s in self.selected_sensors}

        for i in range(n):
            z_scores = {
                s: float(engine_df.iloc[i][z_cols[s]])
                for s in self.selected_sensors
            }

            # Update persistence buffers
            pers_flags = {}
            for s, z_val in z_scores.items():
                buf = z_history[s]
                buf.append(z_val)
                if len(buf) > PERSISTENCE_WINDOW:
                    buf.pop(0)
                is_persistent = (
                    len(buf) >= PERSISTENCE_WINDOW
                    and all(abs(v) >= ANOMALY_THRESHOLD_Z for v in buf)
                )
                pers_flags[f"persistent_anomaly_{s}"] = 1 if is_persistent else 0

            a_score, a_level, severity, evidence = self._compute_anomaly_score(z_scores)
            anomaly_scores[i] = a_score
            anomaly_levels.append(a_level)
            severities.append(severity)
            evidences[i] = evidence
            persistent_flags_list.append(pers_flags)

        result = engine_df.copy()
        result["anomaly_score"] = anomaly_scores
        result["anomaly_level"] = anomaly_levels
        result["severity"] = severities
        result["evidence_score"] = evidences
        # FD001 has a single fault mode: HPC degradation
        # We do NOT map this to AeroVigil's piston-engine fault types.
        result["fault_type"] = "HPC_DEGRADATION"
        # timestamp_sec equivalent: use cycle as ordinal time axis
        result["timestamp_sec"] = result["cycle"].astype(float)

        # Build diagnostic rows for EngineHealthIndex (reuses existing module unchanged)
        health_calc = EngineHealthIndex()

        health_indices = np.zeros(n)
        health_states = []
        health_penalties = np.zeros(n)
        health_trends = []

        for i in range(n):
            diag_point = {
                "anomaly_score": float(anomaly_scores[i]),
                "severity": severities[i],
                "fault_type": "UNKNOWN_ANOMALY" if severities[i] != "NORMAL" else "NORMAL",
                "evidence_score": float(evidences[i]),
                "persistent_flags": persistent_flags_list[i],
            }
            h_res = health_calc.calculate_point(
                diag_point, timestamp_sec=float(result.iloc[i]["cycle"])
            )
            health_indices[i] = h_res["health_index"]
            health_states.append(h_res["health_state"])
            health_penalties[i] = h_res["health_penalty"]
            health_trends.append(h_res["health_trend"])

        result["health_index"] = health_indices
        result["health_state"] = health_states
        result["health_penalty"] = health_penalties
        result["health_trend"] = health_trends

        return result

    def compute_fleet_health_trajectories(
        self, transformed_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes health trajectories for ALL engines in a DataFrame.

        Parameters
        ----------
        transformed_df : pd.DataFrame
            Output of CMAPSSPreprocessor.transform() for all engines.

        Returns
        -------
        pd.DataFrame
            All engines concatenated with health columns appended.
        """
        result_parts = []
        engine_ids = sorted(transformed_df["unit_number"].unique())
        logger.info(
            "Computing health trajectories for %d engines...", len(engine_ids)
        )

        for eng_id in engine_ids:
            eng_df = transformed_df[transformed_df["unit_number"] == eng_id].copy()
            eng_result = self.compute_engine_health_trajectory(eng_df)
            result_parts.append(eng_result)

        result = pd.concat(result_parts, ignore_index=True)
        logger.info("Health trajectories complete. Total rows: %d", len(result))
        return result
