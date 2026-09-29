"""
AeroVigil DT - Phase 4 Health Index, RUL & Mission Risk Automated Test Suite
Verifies Health Index bounds, RUL estimation trajectory logic, mission risk assessment,
flight phase sensitivity, advisory recommendations, and integrated pipeline execution.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure repo root is on path if running as standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator import generate_telemetry
from src.digital_twin import DigitalTwin
from src.residual_analysis import ResidualAnalyzer
from src.health_index import EngineHealthIndex
from src.rul_estimator import RULEstimator
from src.mission_risk import MissionRiskEstimator
from src.health_risk_pipeline import AeroVigilPipeline, run_pipeline
import config


# -----------------------------------------------------------------------------
# 1. Health Index Tests
# -----------------------------------------------------------------------------

def test_health_index_normal():
    """Verify normal healthy telemetry maintains high Health Index (>= 90)."""
    df_raw = generate_telemetry(scenario="normal", duration_minutes=10.0, seed=42)
    twin = DigitalTwin()
    analyzer = ResidualAnalyzer()
    df_diag = analyzer.analyze(twin.predict_expected_state(df_raw))

    health_calc = EngineHealthIndex()
    df_health = health_calc.calculate(df_diag)

    assert (df_health["health_index"] >= 90.0).all()
    assert (df_health["health_state"] == "HEALTHY").all()
    assert (df_health["health_index"] <= 100.0).all()
    assert (df_health["health_index"] >= 0.0).all()


def test_health_index_degradation():
    """Verify overheating scenario significantly lowers Health Index."""
    df_raw = generate_telemetry(scenario="overheating", duration_minutes=20.0, seed=42)
    twin = DigitalTwin()
    analyzer = ResidualAnalyzer()
    df_diag = analyzer.analyze(twin.predict_expected_state(df_raw))

    health_calc = EngineHealthIndex()
    df_health = health_calc.calculate(df_diag)

    final_health = df_health["health_index"].iloc[-1]
    final_state = df_health["health_state"].iloc[-1]

    assert final_health < 30.0
    assert final_state in ["SEVERE", "CRITICAL"]


def test_health_index_bounds_and_determinism():
    """Verify Health Index is strictly bounded within [0, 100] and deterministic."""
    df_raw = generate_telemetry(scenario="lubrication_fault", seed=42)
    twin = DigitalTwin()
    analyzer = ResidualAnalyzer()
    df_diag = analyzer.analyze(twin.predict_expected_state(df_raw))

    calc1 = EngineHealthIndex()
    res1 = calc1.calculate(df_diag)

    calc2 = EngineHealthIndex()
    res2 = calc2.calculate(df_diag)

    assert (res1["health_index"] >= 0.0).all()
    assert (res1["health_index"] <= 100.0).all()
    pd.testing.assert_frame_equal(res1, res2)


# -----------------------------------------------------------------------------
# 2. RUL Estimator Tests
# -----------------------------------------------------------------------------

def test_rul_stable_healthy():
    """Verify healthy telemetry produces STABLE or UNAVAILABLE RUL status without fabricated drops."""
    df_raw = generate_telemetry(scenario="normal", seed=42)
    df_health = EngineHealthIndex().calculate(ResidualAnalyzer().analyze(DigitalTwin().predict_expected_state(df_raw)))

    rul_est = RULEstimator()
    df_rul = rul_est.estimate(df_health)

    assert df_rul["rul_status"].isin(["STABLE", "UNAVAILABLE"]).all()
    assert df_rul["rul_minutes"].isna().all()


def test_rul_active_degradation():
    """Verify active health degradation produces decreasing non-negative RUL estimate."""
    df_raw = generate_telemetry(scenario="overheating", duration_minutes=20.0, seed=42)
    pipeline = AeroVigilPipeline()
    df_pipe = pipeline.process(df_raw)

    degrading_rows = df_pipe[df_pipe["rul_status"].isin(["DEGRADING", "LOW_RUL", "CRITICAL_RUL"])]
    assert not degrading_rows.empty

    valid_ruls = degrading_rows["rul_seconds"].dropna()
    assert (valid_ruls >= 0.0).all()


# -----------------------------------------------------------------------------
# 3. Mission Risk Model Tests
# -----------------------------------------------------------------------------

def test_mission_risk_normal():
    """Verify normal telemetry yields LOW risk and CONTINUE_MONITORING advisory."""
    df_raw = generate_telemetry(scenario="normal", seed=42)
    df_pipe = run_pipeline(df_raw)

    final_row = df_pipe.iloc[-1]
    assert final_row["mission_risk_level"] == "LOW"
    assert final_row["mission_recommendation"] == "CONTINUE_MONITORING"
    assert final_row["mission_risk_score"] < 20.0


def test_mission_risk_critical_fault():
    """Verify severe thermal/lubrication fault yields CRITICAL risk and CONSIDER_MISSION_ABORT advisory."""
    df_raw = generate_telemetry(scenario="lubrication_fault", seed=42)
    df_pipe = run_pipeline(df_raw)

    final_row = df_pipe.iloc[-1]
    assert final_row["mission_risk_level"] == "CRITICAL"
    assert final_row["mission_recommendation"] == "CONSIDER_MISSION_ABORT"
    assert final_row["mission_risk_score"] >= 70.0


def test_mission_risk_sensor_drift_advisory():
    """Verify sensor drift yields MODERATE risk and INSPECT_AT_NEXT_OPPORTUNITY advisory."""
    df_raw = generate_telemetry(scenario="sensor_drift", drift_sensor="CHT", seed=42)
    df_pipe = run_pipeline(df_raw)

    final_row = df_pipe.iloc[-1]
    assert final_row["mission_risk_level"] == "MODERATE"
    assert final_row["mission_recommendation"] == "INSPECT_AT_NEXT_OPPORTUNITY"


def test_mission_phase_sensitivity():
    """Verify TAKEOFF phase produces higher risk score than CRUISE for identical degradation."""
    risk_est = MissionRiskEstimator()

    res_takeoff = risk_est.assess_point(
        health_index=60.0,
        severity="HIGH",
        fault_type="OVERHEATING",
        rul_minutes=8.0,
        mission_phase="TAKEOFF",
    )

    res_cruise = risk_est.assess_point(
        health_index=60.0,
        severity="HIGH",
        fault_type="OVERHEATING",
        rul_minutes=8.0,
        mission_phase="CRUISE",
    )

    assert res_takeoff["mission_risk_score"] > res_cruise["mission_risk_score"]


# -----------------------------------------------------------------------------
# 4. Integrated Pipeline Tests
# -----------------------------------------------------------------------------

def test_full_integrated_pipeline_schema():
    """Verify end-to-end pipeline execution populates all required Phase 1-4 columns cleanly."""
    df_raw = generate_telemetry(scenario="normal", duration_minutes=5.0, seed=42)
    df_out = run_pipeline(df_raw)

    required_phase4_cols = [
        "health_index",
        "health_state",
        "health_penalty",
        "health_trend",
        "degradation_rate",
        "rul_seconds",
        "rul_minutes",
        "rul_status",
        "mission_risk_score",
        "mission_risk_level",
        "mission_recommendation",
    ]

    for col in required_phase4_cols:
        assert col in df_out.columns, f"Missing Phase 4 column: {col}"

    assert len(df_out) == 300
    assert not df_out["health_index"].isna().any()
    assert not df_out["mission_risk_score"].isna().any()
