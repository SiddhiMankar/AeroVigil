"""
AeroVigil DT - Mission Risk Model & Advisory Component
Phase 4 Component

Evaluates operational mission risk and provides prototype decision advisories
based on Engine Health Index, RUL estimates, fault severity, and flight phase sensitivity.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

import sys
import os

# Ensure repo root is on path if running as standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.rul_estimator import RULEstimator


class MissionRiskEstimator:
    """
    Phase-Aware Mission Risk Estimator & Advisory System.

    Combines engine health index, remaining useful life estimates, fault classification,
    severity levels, and mission phase vulnerability multipliers to assess mission risk (0-100)
    and output prototype operational advisories.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets internal state variables."""
        pass

    def _determine_risk_level(self, risk_score: float) -> str:
        """Maps mission risk score to risk level category."""
        if risk_score < config.RISK_LEVEL_THRESHOLDS["LOW"]:
            return "LOW"
        elif risk_score < config.RISK_LEVEL_THRESHOLDS["MODERATE"]:
            return "MODERATE"
        elif risk_score < config.RISK_LEVEL_THRESHOLDS["HIGH"]:
            return "HIGH"
        else:
            return "CRITICAL"

    def _determine_recommendation(self, risk_level: str, fault_type: str) -> str:
        """Determines prototype operational mission advisory."""
        if fault_type == "SENSOR_DRIFT" and risk_level != "CRITICAL":
            return "INSPECT_AT_NEXT_OPPORTUNITY"

        if risk_level == "LOW":
            return "CONTINUE_MONITORING"
        elif risk_level == "MODERATE":
            return "INCREASE_MONITORING"
        elif risk_level == "HIGH":
            return "INSPECT_AT_NEXT_OPPORTUNITY"
        else:
            return "CONSIDER_MISSION_ABORT"

    def assess_point(
        self,
        health_index: float,
        severity: str = "NORMAL",
        fault_type: str = "NORMAL",
        rul_minutes: Optional[float] = None,
        mission_phase: str = "CRUISE",
    ) -> Dict[str, Union[float, str]]:
        """
        Assesses mission risk for a single operational state point.

        Parameters:
        -----------
        health_index : float
            Current Engine Health Index value (0.0 to 100.0).
        severity : str
            Diagnostic severity level ('NORMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
        fault_type : str
            Classified fault type.
        rul_minutes : Optional[float]
            Estimated RUL in minutes (or NaN if unavailable).
        mission_phase : str
            Active flight phase ('STARTUP', 'TAKEOFF', 'CLIMB', 'CRUISE', 'MANEUVER', 'DESCENT', 'LANDING').

        Returns:
        --------
        Dict containing mission_risk_score, mission_risk_level, and mission_recommendation.
        """
        # 1. Base Health Component
        r_health = max(0.0, 100.0 - float(health_index))

        # 2. Severity Component
        r_severity = config.SEVERITY_PENALTY_MAP.get(severity, 0.0)

        # 3. RUL Component
        if rul_minutes is None or np.isnan(rul_minutes) or float(rul_minutes) >= 30.0:
            r_rul = 0.0
        else:
            r_rul = min(100.0, max(0.0, (30.0 - float(rul_minutes)) * 3.33))

        # 4. Fault Type Specific Multiplier
        fault_factors = {
            "NORMAL": 0.0,
            "SENSOR_DRIFT": 0.55,       # Sensor drift has lower risk because physical engine is healthy
            "VIBRATION_ANOMALY": 1.05,
            "OVERHEATING": 1.30,        # Severe thermal threat
            "LUBRICATION_FAULT": 1.40,  # Critical hydraulic threat
            "UNKNOWN_ANOMALY": 1.00,
        }
        fault_factor = fault_factors.get(fault_type, 1.0)

        # 5. Mission Phase Multiplier
        phase_mult = config.MISSION_PHASE_MULTIPLIERS.get(mission_phase, 1.0)

        # Combined Composite Risk Calculation
        weighted_base = 0.35 * r_health + 0.40 * r_severity + 0.25 * r_rul
        if fault_type == "NORMAL":
            raw_risk = weighted_base * phase_mult
        else:
            raw_risk = weighted_base * phase_mult * fault_factor

        risk_score = round(float(min(100.0, max(0.0, raw_risk))), 1)
        risk_level = self._determine_risk_level(risk_score)
        recommendation = self._determine_recommendation(risk_level, fault_type)

        return {
            "mission_risk_score": risk_score,
            "mission_risk_level": risk_level,
            "mission_recommendation": recommendation,
        }

    def assess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch Mission Risk assessment for a DataFrame.
        If Phase 4 RUL estimator columns are missing, runs RULEstimator first automatically.

        Parameters:
        -----------
        df : pd.DataFrame
            Telemetry dataframe.

        Returns:
        --------
        pd.DataFrame
            Augmented DataFrame containing mission_risk_score, mission_risk_level, and mission_recommendation.
        """
        self.reset()
        result_df = df.copy()

        # Run RULEstimator if Phase 4 columns are missing
        if "rul_minutes" not in result_df.columns:
            rul_est = RULEstimator()
            result_df = rul_est.estimate(result_df)

        n_rows = len(result_df)

        risk_scores = np.zeros(n_rows)
        risk_levels = []
        recommendations = []

        health_vals = result_df["health_index"].values
        severities = result_df["severity"].values
        faults = result_df["fault_type"].values
        rul_mins = result_df["rul_minutes"].values
        phases = result_df["mission_phase"].values if "mission_phase" in result_df.columns else ["CRUISE"] * n_rows

        for i in range(n_rows):
            h_val = float(health_vals[i])
            sev = str(severities[i])
            flt = str(faults[i])
            rul_m = float(rul_mins[i]) if not np.isnan(rul_mins[i]) else None
            ph = str(phases[i])

            res = self.assess_point(
                health_index=h_val,
                severity=sev,
                fault_type=flt,
                rul_minutes=rul_m,
                mission_phase=ph,
            )

            risk_scores[i] = res["mission_risk_score"]
            risk_levels.append(res["mission_risk_level"])
            recommendations.append(res["mission_recommendation"])

        result_df["mission_risk_score"] = risk_scores
        result_df["mission_risk_level"] = risk_levels
        result_df["mission_recommendation"] = recommendations

        return result_df
