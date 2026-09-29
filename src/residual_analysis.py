"""
AeroVigil DT - Residual Analysis & Anomaly Detection Module
Phase 3 Foundation Component

Transforms Digital Twin normalized residuals into explainable anomaly scores,
persistence flags, rule-based fault classifications, contributing signal explanations,
and evidence scores.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import sys
import os

# Ensure repo root is on path if running as standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.digital_twin import DigitalTwin


class ResidualAnalyzer:
    """
    Explainable Physics-Informed Residual Analyzer & Fault Classifier.

    Evaluates normalized residuals (Observed - Expected) / Nominal_StdDev across time steps.
    Applies multi-signal scoring, temporal persistence, rule-based fault classification,
    and transparent evidence score calculations.
    """

    def __init__(
        self,
        persistence_window: int = config.DEFAULT_PERSISTENCE_WINDOW,
        warning_threshold: float = config.WARNING_THRESHOLD_STD,
        anomaly_threshold: float = config.ANOMALY_THRESHOLD_STD,
        critical_threshold: float = config.CRITICAL_THRESHOLD_STD,
    ):
        self.persistence_window = persistence_window
        self.warning_threshold = warning_threshold
        self.anomaly_threshold = anomaly_threshold
        self.critical_threshold = critical_threshold
        self.reset()

    def reset(self):
        """Resets internal sliding window history for online detection."""
        self.history: Dict[str, List[float]] = {sig: [] for sig in config.MODELED_SIGNALS}
        self.step_counter: int = 0

    def _update_history_and_check_persistence(self, sig: str, norm_res: float) -> Tuple[bool, bool]:
        """
        Updates sliding history for a signal and returns (instant_anomaly, persistent_anomaly).
        """
        instant_anomaly = abs(norm_res) >= self.anomaly_threshold

        buf = self.history[sig]
        buf.append(norm_res)
        if len(buf) > self.persistence_window:
            buf.pop(0)

        # Persistent anomaly requires at least persistence_window consecutive samples >= threshold
        if len(buf) >= self.persistence_window and all(abs(val) >= self.anomaly_threshold for val in buf):
            persistent_anomaly = True
        else:
            persistent_anomaly = False

        return instant_anomaly, persistent_anomaly

    def analyze_point(
        self,
        normalized_residuals: Dict[str, float],
        raw_residuals: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Union[float, str, List[str], Dict[str, int]]]:
        """
        Processes a single time-step normalized residual dictionary.

        Parameters:
        -----------
        normalized_residuals : Dict[str, float]
            Dictionary of normalized residual values per signal.
        raw_residuals : Optional[Dict[str, float]]
            Optional dictionary of raw physical residuals (e.g. °C, psi).

        Returns:
        --------
        Dict containing diagnostic classification, scores, and explanations.
        """
        self.step_counter += 1

        instant_flags = {}
        persistent_flags = {}
        contributing_signals = []
        reasoning_items = []

        # 1. Update history & check per-signal anomalies
        for sig in config.MODELED_SIGNALS:
            z = normalized_residuals.get(sig, 0.0)
            inst_flag, pers_flag = self._update_history_and_check_persistence(sig, z)
            instant_flags[f"anomaly_{sig}"] = 1 if inst_flag else 0
            persistent_flags[f"persistent_anomaly_{sig}"] = 1 if pers_flag else 0

            if abs(z) >= self.warning_threshold:
                contributing_signals.append(sig)
                sign_str = "+" if z >= 0 else ""
                reasoning_items.append(f"{sig} residual {sign_str}{z:.1f}σ")

        # 2. Global Anomaly Score (0.0 to 100.0)
        weighted_sum = 0.0
        total_weight = 0.0
        for sig, weight in config.SIGNAL_WEIGHTS.items():
            z = abs(normalized_residuals.get(sig, 0.0))
            if z >= self.warning_threshold:
                excess = z - (self.warning_threshold - 0.5)
                weighted_sum += weight * (excess ** 1.3)
            total_weight += weight

        raw_score = (weighted_sum / total_weight) * 22.0
        anomaly_score = round(float(min(100.0, max(0.0, raw_score))), 1)

        # Anomaly level category
        if anomaly_score < 15.0:
            anomaly_level = "NORMAL"
        elif anomaly_score < 35.0:
            anomaly_level = "WARNING"
        elif anomaly_score < 65.0:
            anomaly_level = "ANOMALOUS"
        else:
            anomaly_level = "CRITICAL"

        # 3. Rule-Based Fault Classification
        z_cht = normalized_residuals.get("CHT", 0.0)
        z_egt = normalized_residuals.get("EGT", 0.0)
        z_oil_p = normalized_residuals.get("oil_pressure", 0.0)
        z_oil_t = normalized_residuals.get("oil_temperature", 0.0)
        z_vib = normalized_residuals.get("vibration", 0.0)
        z_rpm = normalized_residuals.get("RPM", 0.0)

        fault_type = "NORMAL"
        evidence_score = 0.0

        # Check Overheating: CHT strongly positive AND EGT strongly positive
        if z_cht >= 2.5 and z_egt >= 2.5:
            fault_type = "OVERHEATING"
            # Evidence based on CHT/EGT severity and persistence
            pers_match = 1.0 if (persistent_flags["persistent_anomaly_CHT"] or persistent_flags["persistent_anomaly_EGT"]) else 0.75
            evidence_score = round(min(0.98, 0.60 + 0.03 * min(12.0, (z_cht + z_egt) / 2.0)) * pers_match, 2)

        # Check Lubrication Fault: Oil Pressure strongly negative AND Oil Temp positive
        elif z_oil_p <= -2.5 and z_oil_t >= 1.5:
            fault_type = "LUBRICATION_FAULT"
            pers_match = 1.0 if (persistent_flags["persistent_anomaly_oil_pressure"] or persistent_flags["persistent_anomaly_oil_temperature"]) else 0.75
            evidence_score = round(min(0.98, 0.60 + 0.03 * min(12.0, abs(z_oil_p) + z_oil_t)) * pers_match, 2)

        # Check Vibration Anomaly: Vibration strongly positive
        elif z_vib >= 2.5:
            fault_type = "VIBRATION_ANOMALY"
            pers_match = 1.0 if persistent_flags["persistent_anomaly_vibration"] else 0.75
            evidence_score = round(min(0.98, 0.65 + 0.04 * min(10.0, z_vib)) * pers_match, 2)

        # Check Sensor Drift: EXACTLY ONE primary thermal/hydraulic sensor strongly abnormal while correlated physics nominal
        else:
            primary_signals = ["CHT", "EGT", "oil_pressure", "oil_temperature"]
            abnormal_primaries = [sig for sig in primary_signals if abs(normalized_residuals.get(sig, 0.0)) >= self.anomaly_threshold]

            if len(abnormal_primaries) == 1:
                target_sig = abnormal_primaries[0]
                other_primaries = [sig for sig in primary_signals if sig != target_sig]
                others_nominal = all(abs(normalized_residuals.get(sig, 0.0)) < self.warning_threshold for sig in other_primaries)

                if others_nominal:
                    fault_type = "SENSOR_DRIFT"
                    z_target = abs(normalized_residuals.get(target_sig, 0.0))
                    evidence_score = round(min(0.95, 0.70 + 0.03 * min(8.0, z_target)), 2)
            
            # Check if any signal has breached anomaly threshold but pattern doesn't match known rules
            if fault_type == "NORMAL":
                any_anomalous = any(abs(z) >= self.anomaly_threshold for z in normalized_residuals.values())
                if any_anomalous or anomaly_score >= 30.0:
                    fault_type = "UNKNOWN_ANOMALY"
                    max_z = max(abs(z) for z in normalized_residuals.values())
                    evidence_score = round(min(0.85, 0.40 + 0.04 * min(10.0, max_z)), 2)

        # 4. Severity Assessment
        max_abs_z = max([abs(normalized_residuals.get(sig, 0.0)) for sig in config.MODELED_SIGNALS] + [0.0])
        any_persistent = any(persistent_flags.values())

        if fault_type == "NORMAL" and max_abs_z < self.warning_threshold:
            severity = "NORMAL"
            evidence_score = 0.0
        elif max_abs_z < self.anomaly_threshold and not any_persistent:
            severity = "LOW"
        elif fault_type == "SENSOR_DRIFT" or max_abs_z < self.critical_threshold:
            severity = "MEDIUM" if not any_persistent else "HIGH"
        elif any_persistent and (fault_type in ["OVERHEATING", "LUBRICATION_FAULT"] or max_abs_z >= 6.0):
            severity = "CRITICAL"
        else:
            severity = "HIGH"

        reasoning_str = "; ".join(reasoning_items) if reasoning_items else "All signal residuals nominal within ±2.0σ."

        return {
            "anomaly_score": anomaly_score,
            "anomaly_level": anomaly_level,
            "fault_type": fault_type,
            "severity": severity,
            "evidence_score": evidence_score,
            "contributing_signals": contributing_signals,
            "reasoning": reasoning_str,
            "instant_flags": instant_flags,
            "persistent_flags": persistent_flags,
        }

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch analysis of an augmented telemetry DataFrame (containing Digital Twin residuals).
        If Digital Twin columns do not exist, runs DigitalTwin first automatically.

        Parameters:
        -----------
        df : pd.DataFrame
            Telemetry dataframe.

        Returns:
        --------
        pd.DataFrame
            Augmented DataFrame containing anomaly scores, flags, fault classification,
            severity, evidence scores, and contributing signal explanations.
        """
        self.reset()
        result_df = df.copy()

        # If expected/residual columns do not exist, run DigitalTwin first
        if "normalized_residual_CHT" not in result_df.columns:
            twin = DigitalTwin()
            result_df = twin.predict_expected_state(result_df)

        n_rows = len(result_df)

        scores = np.zeros(n_rows)
        levels = []
        faults = []
        severities = []
        evidences = np.zeros(n_rows)
        contrib_list = []
        reasoning_list = []

        instant_cols = {f"anomaly_{sig}": np.zeros(n_rows, dtype=int) for sig in config.MODELED_SIGNALS}
        persistent_cols = {f"persistent_anomaly_{sig}": np.zeros(n_rows, dtype=int) for sig in config.MODELED_SIGNALS}

        for i in range(n_rows):
            norm_res = {sig: result_df[f"normalized_residual_{sig}"].iloc[i] for sig in config.MODELED_SIGNALS}
            raw_res = {sig: result_df[f"residual_{sig}"].iloc[i] for sig in config.MODELED_SIGNALS}

            res = self.analyze_point(normalized_residuals=norm_res, raw_residuals=raw_res)

            scores[i] = res["anomaly_score"]
            levels.append(res["anomaly_level"])
            faults.append(res["fault_type"])
            severities.append(res["severity"])
            evidences[i] = res["evidence_score"]
            contrib_list.append(", ".join(res["contributing_signals"]))
            reasoning_list.append(res["reasoning"])

            for sig in config.MODELED_SIGNALS:
                instant_cols[f"anomaly_{sig}"][i] = res["instant_flags"][f"anomaly_{sig}"]
                persistent_cols[f"persistent_anomaly_{sig}"][i] = res["persistent_flags"][f"persistent_anomaly_{sig}"]

        # Append diagnostic results to DataFrame
        result_df["anomaly_score"] = scores
        result_df["anomaly_level"] = levels
        result_df["fault_type"] = faults
        result_df["severity"] = severities
        result_df["evidence_score"] = evidences
        result_df["contributing_signals"] = contrib_list
        result_df["reasoning"] = reasoning_list

        for col_name, arr in instant_cols.items():
            result_df[col_name] = arr
        for col_name, arr in persistent_cols.items():
            result_df[col_name] = arr

        return result_df
