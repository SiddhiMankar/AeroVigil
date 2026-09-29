"""
AeroVigil DT - Dashboard Utility & Data Preparation Module
Phase 5 Component

Provides scenario mapping, cached simulation execution, current-state extraction,
formatting helpers, and scenario comparison matrices for the Streamlit interface.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import sys
import os

# Ensure repo root is on path if running as standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator import generate_telemetry
from src.health_risk_pipeline import run_pipeline
import config

SCENARIO_DISPLAY_NAMES: Dict[str, str] = {
    "normal": "Normal Baseline",
    "overheating": "Overheating Thermal Fault",
    "lubrication_fault": "Lubrication System Fault",
    "vibration_anomaly": "Vibration / Mechanical Anomaly",
    "sensor_drift": "CHT Sensor Drift Bias",
}

SCENARIO_NAME_MAP: Dict[str, str] = {v: k for k, v in SCENARIO_DISPLAY_NAMES.items()}


def run_cached_simulation(
    scenario: str,
    duration_minutes: float = 20.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Executes telemetry generation and full AeroVigil DT pipeline processing.
    """
    df_raw = generate_telemetry(
        scenario=scenario,
        duration_minutes=duration_minutes,
        sample_rate_hz=1.0,
        seed=seed,
    )
    df_processed = run_pipeline(df_raw)
    return df_processed


def extract_snapshot_at_index(df: pd.DataFrame, idx: int) -> Dict[str, Union[float, str]]:
    """
    Extracts complete telemetry, Digital Twin, diagnostic, health, RUL, and risk state
    at a specific row index.
    """
    safe_idx = max(0, min(len(df) - 1, idx))
    row = df.iloc[safe_idx]

    rul_val = row["rul_minutes"]
    if pd.isna(rul_val) or rul_val is None:
        rul_display = "--"
        rul_status_display = row.get("rul_status", "STABLE")
    else:
        rul_display = f"{float(rul_val):.1f} min"
        rul_status_display = row.get("rul_status", "DEGRADING")

    ev_val = row.get("evidence_score", 0.0)
    ev_display = f"{float(ev_val) * 100.0:.0f}%" if ev_val > 0 else "--"

    snapshot = {
        "index": safe_idx,
        "timestamp": str(row.get("timestamp", "")),
        "timestamp_sec": float(row.get("timestamp_sec", 0.0)),
        "mission_phase": str(row.get("mission_phase", "CRUISE")),
        "throttle": float(row.get("throttle", 0.0)),
        "ambient_temperature": float(row.get("ambient_temperature", 0.0)),
        # Raw Telemetry
        "RPM": float(row.get("RPM", 0.0)),
        "CHT": float(row.get("CHT", 0.0)),
        "EGT": float(row.get("EGT", 0.0)),
        "oil_pressure": float(row.get("oil_pressure", 0.0)),
        "oil_temperature": float(row.get("oil_temperature", 0.0)),
        "fuel_flow": float(row.get("fuel_flow", 0.0)),
        "vibration": float(row.get("vibration", 0.0)),
        "battery_voltage": float(row.get("battery_voltage", 0.0)),
        # Expected Telemetry
        "expected_RPM": float(row.get("expected_RPM", 0.0)),
        "expected_CHT": float(row.get("expected_CHT", 0.0)),
        "expected_EGT": float(row.get("expected_EGT", 0.0)),
        "expected_oil_pressure": float(row.get("expected_oil_pressure", 0.0)),
        "expected_oil_temperature": float(row.get("expected_oil_temperature", 0.0)),
        "expected_fuel_flow": float(row.get("expected_fuel_flow", 0.0)),
        "expected_vibration": float(row.get("expected_vibration", 0.0)),
        "expected_battery_voltage": float(row.get("expected_battery_voltage", 0.0)),
        # Residuals
        "residual_CHT": float(row.get("residual_CHT", 0.0)),
        "residual_EGT": float(row.get("residual_EGT", 0.0)),
        "residual_oil_pressure": float(row.get("residual_oil_pressure", 0.0)),
        "normalized_residual_CHT": float(row.get("normalized_residual_CHT", 0.0)),
        "normalized_residual_EGT": float(row.get("normalized_residual_EGT", 0.0)),
        "normalized_residual_oil_pressure": float(row.get("normalized_residual_oil_pressure", 0.0)),
        # Diagnostics
        "anomaly_score": float(row.get("anomaly_score", 0.0)),
        "anomaly_level": str(row.get("anomaly_level", "NORMAL")),
        "fault_type": str(row.get("fault_type", "NORMAL")),
        "severity": str(row.get("severity", "NORMAL")),
        "evidence_score": float(ev_val),
        "evidence_display": ev_display,
        "contributing_signals": str(row.get("contributing_signals", "")),
        "reasoning": str(row.get("reasoning", "")),
        # Health & RUL
        "health_index": float(row.get("health_index", 100.0)),
        "health_state": str(row.get("health_state", "HEALTHY")),
        "health_trend": str(row.get("health_trend", "STABLE")),
        "degradation_rate": float(row.get("degradation_rate", 0.0)),
        "rul_seconds": float(row.get("rul_seconds", np.nan)),
        "rul_minutes": float(row.get("rul_minutes", np.nan)),
        "rul_display": rul_display,
        "rul_status": rul_status_display,
        # Risk & Advisory
        "mission_risk_score": float(row.get("mission_risk_score", 0.0)),
        "mission_risk_level": str(row.get("mission_risk_level", "LOW")),
        "mission_recommendation": str(row.get("mission_recommendation", "CONTINUE_MONITORING")),
    }
    return snapshot


def generate_scenario_comparison(duration_minutes: float = 20.0, seed: int = 42) -> pd.DataFrame:
    """
    Runs the pipeline across all 5 scenarios and returns a comparison summary table.
    """
    scenarios = ["normal", "overheating", "lubrication_fault", "vibration_anomaly", "sensor_drift"]
    matrix_rows = []

    for sc in scenarios:
        df = run_cached_simulation(scenario=sc, duration_minutes=duration_minutes, seed=seed)
        final_row = df.iloc[-1]

        min_health = df["health_index"].min()
        valid_ruls = df["rul_minutes"].dropna()
        min_rul_val = valid_ruls.min() if not valid_ruls.empty else None

        matrix_rows.append(
            {
                "Scenario Key": sc,
                "Scenario": SCENARIO_DISPLAY_NAMES.get(sc, sc),
                "Min Health": round(float(min_health), 1),
                "Final Health": round(float(final_row["health_index"]), 1),
                "Final Health State": str(final_row["health_state"]),
                "Min RUL (min)": f"{min_rul_val:.1f}" if min_rul_val is not None else "--",
                "Final Risk Score": round(float(final_row["mission_risk_score"]), 1),
                "Final Risk Level": str(final_row["mission_risk_level"]),
                "Detected Fault": str(final_row["fault_type"]),
                "Final Advisory": str(final_row["mission_recommendation"]),
            }
        )

    return pd.DataFrame(matrix_rows)
