"""
AeroVigil DT - Engine Health Index Module
Phase 4 Foundation Component

Computes a normalized, transparent, physics-informed Engine Health Index (0 to 100)
from Digital Twin residual diagnostic states.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

import sys
import os

# Ensure repo root is on path if running as standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.residual_analysis import ResidualAnalyzer


class EngineHealthIndex:
    """
    Transparent Physics-Informed Engine Health Index Calculator.

    Transforms diagnostic states (anomaly score, severity, persistence, fault type)
    into a continuous health index (0 = CRITICAL, 100 = HEALTHY) with exponential smoothing
    and health state categorization.
    """

    def __init__(self, smoothing_alpha: float = config.HEALTH_SMOOTHING_ALPHA):
        self.smoothing_alpha = smoothing_alpha
        self.reset()

    def reset(self):
        """Resets internal state variables of the smoothing filter and history."""
        self.curr_health: float = 100.0
        self.health_history: List[float] = []
        self.time_history: List[float] = []
        self.step_counter: int = 0

    def _determine_health_state(self, health_idx: float) -> str:
        """Maps health index numerical score to categorical health state."""
        if health_idx >= config.HEALTH_STATE_THRESHOLDS["HEALTHY"]:
            return "HEALTHY"
        elif health_idx >= config.HEALTH_STATE_THRESHOLDS["DEGRADED"]:
            return "DEGRADED"
        elif health_idx >= config.HEALTH_STATE_THRESHOLDS["WARNING"]:
            return "WARNING"
        elif health_idx >= config.HEALTH_STATE_THRESHOLDS["SEVERE"]:
            return "SEVERE"
        else:
            return "CRITICAL"

    def calculate_point(
        self,
        diagnostic_point: Dict[str, Union[float, str, List[str], Dict[str, int]]],
        timestamp_sec: float = 0.0,
    ) -> Dict[str, Union[float, str]]:
        """
        Calculates Health Index for a single time-step diagnostic output.

        Parameters:
        -----------
        diagnostic_point : Dict
            Output dictionary from ResidualAnalyzer.analyze_point() or DataFrame row.
        timestamp_sec : float
            Current time in elapsed flight seconds.

        Returns:
        --------
        Dict containing health_index, health_state, health_penalty, and health_trend.
        """
        anomaly_score = float(diagnostic_point.get("anomaly_score", 0.0))
        severity = str(diagnostic_point.get("severity", "NORMAL"))
        fault_type = str(diagnostic_point.get("fault_type", "NORMAL"))
        evidence_score = float(diagnostic_point.get("evidence_score", 0.0))

        # Count active persistent anomaly signals
        pers_flags = diagnostic_point.get("persistent_flags", {})
        num_persistent = sum(1 for v in pers_flags.values() if v > 0)

        # 1. Component Penalties
        p_anom = anomaly_score
        p_sev = config.SEVERITY_PENALTY_MAP.get(severity, 0.0)
        p_flt = config.FAULT_TYPE_PENALTY_MAP.get(fault_type, 0.0) * evidence_score
        p_pers = min(100.0, num_persistent * 35.0)

        w = config.HEALTH_PENALTY_WEIGHTS
        raw_penalty = (
            w["anomaly_score"] * p_anom
            + w["severity"] * p_sev
            + w["fault_type"] * p_flt
            + w["persistence"] * p_pers
        )

        instant_health = max(0.0, min(100.0, 100.0 - raw_penalty))

        # 2. Smooth Exponential Moving Average
        if self.step_counter == 0:
            self.curr_health = instant_health
        else:
            self.curr_health += self.smoothing_alpha * (instant_health - self.curr_health)

        health_val = round(float(self.curr_health), 1)
        penalty_val = round(float(100.0 - health_val), 1)
        health_state = self._determine_health_state(health_val)

        # 3. Health Trend Calculation over short window (last 10 seconds)
        self.health_history.append(health_val)
        self.time_history.append(timestamp_sec)
        if len(self.health_history) > 15:
            self.health_history.pop(0)
            self.time_history.pop(0)

        if len(self.health_history) >= 5:
            dt_win = self.time_history[-1] - self.time_history[0]
            dh_win = self.health_history[-1] - self.health_history[0]
            if dt_win > 0:
                slope = dh_win / dt_win
                if slope < -0.03:
                    health_trend = "DEGRADING"
                elif slope > 0.03:
                    health_trend = "IMPROVING"
                else:
                    health_trend = "STABLE"
            else:
                health_trend = "STABLE"
        else:
            health_trend = "STABLE"

        self.step_counter += 1

        return {
            "health_index": health_val,
            "health_state": health_state,
            "health_penalty": penalty_val,
            "health_trend": health_trend,
        }

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch Health Index calculation for a diagnostic DataFrame.
        If diagnostic columns do not exist, runs ResidualAnalyzer first automatically.

        Parameters:
        -----------
        df : pd.DataFrame
            Telemetry dataframe.

        Returns:
        --------
        pd.DataFrame
            Augmented DataFrame containing health_index, health_state, health_penalty, and health_trend.
        """
        self.reset()
        result_df = df.copy()

        # Run Phase 3 ResidualAnalyzer if diagnostic columns are missing
        if "fault_type" not in result_df.columns:
            analyzer = ResidualAnalyzer()
            result_df = analyzer.analyze(result_df)

        n_rows = len(result_df)

        health_indices = np.zeros(n_rows)
        health_states = []
        health_penalties = np.zeros(n_rows)
        health_trends = []

        timestamps = (
            result_df["timestamp_sec"].values
            if "timestamp_sec" in result_df.columns
            else np.arange(n_rows, dtype=float)
        )

        for i in range(n_rows):
            row_dict = {
                "anomaly_score": result_df["anomaly_score"].iloc[i],
                "severity": result_df["severity"].iloc[i],
                "fault_type": result_df["fault_type"].iloc[i],
                "evidence_score": result_df["evidence_score"].iloc[i],
                "persistent_flags": {
                    f"persistent_anomaly_{sig}": result_df[f"persistent_anomaly_{sig}"].iloc[i]
                    for sig in config.MODELED_SIGNALS
                    if f"persistent_anomaly_{sig}" in result_df.columns
                },
            }

            res = self.calculate_point(row_dict, timestamp_sec=float(timestamps[i]))

            health_indices[i] = res["health_index"]
            health_states.append(res["health_state"])
            health_penalties[i] = res["health_penalty"]
            health_trends.append(res["health_trend"])

        result_df["health_index"] = health_indices
        result_df["health_state"] = health_states
        result_df["health_penalty"] = health_penalties
        result_df["health_trend"] = health_trends

        return result_df
