"""
AeroVigil DT - Prototype RUL Estimator Module
Phase 4 Foundation Component

Estimates prototype Remaining Useful Life (RUL) in seconds and minutes from health index
degradation trajectories.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

import sys
import os

# Ensure repo root is on path if running as standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


class RULEstimator:
    """
    Prototype Remaining Useful Life (RUL) Estimator.

    Computes recent health degradation slope (dH/dt) over a sliding time window
    and projects remaining time until health reaches the critical threshold (H_crit = 25.0).
    """

    def __init__(
        self,
        window_seconds: float = config.RUL_WINDOW_SECONDS,
        critical_threshold: float = config.CRITICAL_HEALTH_THRESHOLD,
        min_degradation_rate: float = config.MIN_DEGRADATION_RATE,
    ):
        self.window_seconds = window_seconds
        self.critical_threshold = critical_threshold
        self.min_degradation_rate = min_degradation_rate
        self.reset()

    def reset(self):
        """Resets sliding window history."""
        self.health_history: List[float] = []
        self.time_history: List[float] = []
        self.last_rul_seconds: Optional[float] = None

    def estimate_point(
        self, health_index: float, timestamp_sec: float
    ) -> Dict[str, Union[float, str]]:
        """
        Calculates prototype RUL for a single health index observation point.

        Parameters:
        -----------
        health_index : float
            Current Engine Health Index value (0.0 to 100.0).
        timestamp_sec : float
            Current time in elapsed flight seconds.

        Returns:
        --------
        Dict containing degradation_rate, rul_seconds, rul_minutes, and rul_status.
        """
        self.health_history.append(health_index)
        self.time_history.append(timestamp_sec)

        # Retain history strictly within sliding window_seconds
        cutoff_time = timestamp_sec - self.window_seconds
        while len(self.time_history) > 2 and self.time_history[0] < cutoff_time:
            self.time_history.pop(0)
            self.health_history.pop(0)

        # Determine degradation rate via linear slope estimation over window
        n_samples = len(self.health_history)
        dt_win = self.time_history[-1] - self.time_history[0] if n_samples > 1 else 0.0

        if n_samples >= 5 and dt_win >= 10.0:
            # Linear least squares slope dH/dt
            t_arr = np.array(self.time_history) - self.time_history[0]
            h_arr = np.array(self.health_history)
            deg_rate = float(np.polyfit(t_arr, h_arr, 1)[0])
        else:
            deg_rate = 0.0

        deg_rate = round(deg_rate, 4)

        # Evaluate RUL status and remaining time
        if health_index >= 92.0 or deg_rate >= self.min_degradation_rate:
            # Stable healthy state or no sustained degradation
            rul_sec = float("nan")
            rul_min = float("nan")
            deg_rate = max(0.0, deg_rate)
            rul_status = "STABLE" if health_index >= 75.0 else "UNAVAILABLE"
        else:
            # Active health degradation
            remaining_health = max(0.0, health_index - self.critical_threshold)
            abs_deg_rate = abs(deg_rate)

            if abs_deg_rate > 1e-4:
                calc_rul_sec = remaining_health / abs_deg_rate
            else:
                calc_rul_sec = float("nan")

            if np.isnan(calc_rul_sec):
                rul_sec = float("nan")
                rul_min = float("nan")
                rul_status = "UNAVAILABLE"
            else:
                # Exponential smoothing of RUL estimate to avoid jitter
                if self.last_rul_seconds is not None and not np.isnan(self.last_rul_seconds):
                    smoothed_rul = self.last_rul_seconds + 0.20 * (calc_rul_sec - self.last_rul_seconds)
                else:
                    smoothed_rul = calc_rul_sec

                self.last_rul_seconds = smoothed_rul

                rul_sec = round(float(max(0.0, smoothed_rul)), 1)
                rul_min = round(float(max(0.0, smoothed_rul / 60.0)), 2)

                if rul_min < 3.0 or health_index <= self.critical_threshold:
                    rul_status = "CRITICAL_RUL"
                elif rul_min < 10.0:
                    rul_status = "LOW_RUL"
                else:
                    rul_status = "DEGRADING"

        return {
            "degradation_rate": deg_rate,
            "rul_seconds": rul_sec,
            "rul_minutes": rul_min,
            "rul_status": rul_status,
        }

    def estimate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch RUL estimation for a DataFrame containing Health Index values.
        If health_index is missing, runs EngineHealthIndex first automatically.

        Parameters:
        -----------
        df : pd.DataFrame
            Telemetry dataframe.

        Returns:
        --------
        pd.DataFrame
            Augmented DataFrame containing degradation_rate, rul_seconds, rul_minutes, and rul_status.
        """
        self.reset()
        result_df = df.copy()

        # Run Phase 4 EngineHealthIndex if health_index is missing
        if "health_index" not in result_df.columns:
            health_calc = EngineHealthIndex()
            result_df = health_calc.calculate(result_df)

        n_rows = len(result_df)

        deg_rates = np.zeros(n_rows)
        rul_secs = np.full(n_rows, np.nan)
        rul_mins = np.full(n_rows, np.nan)
        rul_statuses = []

        timestamps = (
            result_df["timestamp_sec"].values
            if "timestamp_sec" in result_df.columns
            else np.arange(n_rows, dtype=float)
        )
        health_vals = result_df["health_index"].values

        for i in range(n_rows):
            h_val = float(health_vals[i])
            t_sec = float(timestamps[i])

            res = self.estimate_point(health_index=h_val, timestamp_sec=t_sec)

            deg_rates[i] = res["degradation_rate"]
            rul_secs[i] = res["rul_seconds"]
            rul_mins[i] = res["rul_minutes"]
            rul_statuses.append(res["rul_status"])

        result_df["degradation_rate"] = deg_rates
        result_df["rul_seconds"] = rul_secs
        result_df["rul_minutes"] = rul_mins
        result_df["rul_status"] = rul_statuses

        return result_df
