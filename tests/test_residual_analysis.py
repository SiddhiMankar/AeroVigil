"""
AeroVigil DT - Residual Analysis Automated Test Suite
Verifies anomaly scoring, temporal persistence, rule-based fault classification,
sensor-drift isolation, unknown anomaly handling, and determinism.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure root workspace is in import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator import generate_telemetry
from src.digital_twin import DigitalTwin
from src.residual_analysis import ResidualAnalyzer
import config


def test_normal_telemetry_classification():
    """Test 1: Verify normal healthy telemetry produces NORMAL classification and low score."""
    df_raw = generate_telemetry(scenario="normal", duration_minutes=20.0, seed=42)
    twin = DigitalTwin()
    df_dt = twin.predict_expected_state(df_raw)

    analyzer = ResidualAnalyzer()
    df_diag = analyzer.analyze(df_dt)

    # All rows should be classified as NORMAL fault_type
    assert (df_diag["fault_type"] == "NORMAL").all()
    # Severity under normal operation should stay in NORMAL or LOW
    assert (df_diag["severity"].isin(["NORMAL", "LOW"])).all()
    assert df_diag["anomaly_score"].mean() < 5.0
    assert df_diag["evidence_score"].mean() == 0.0



def test_overheating_classification():
    """Test 2: Verify overheating scenario classifies as OVERHEATING with CHT & EGT contributors."""
    df_raw = generate_telemetry(scenario="overheating", duration_minutes=20.0, seed=42)
    twin = DigitalTwin()
    df_dt = twin.predict_expected_state(df_raw)

    analyzer = ResidualAnalyzer()
    df_diag = analyzer.analyze(df_dt)

    final_row = df_diag.iloc[-1]
    assert final_row["fault_type"] == "OVERHEATING"
    assert final_row["severity"] in ["HIGH", "CRITICAL"]
    assert final_row["evidence_score"] >= 0.80

    contribs = final_row["contributing_signals"]
    assert "CHT" in contribs
    assert "EGT" in contribs


def test_lubrication_fault_classification():
    """Test 3: Verify lubrication fault classifies as LUBRICATION_FAULT with oil_pressure & oil_temperature."""
    df_raw = generate_telemetry(scenario="lubrication_fault", duration_minutes=20.0, seed=42)
    twin = DigitalTwin()
    df_dt = twin.predict_expected_state(df_raw)

    analyzer = ResidualAnalyzer()
    df_diag = analyzer.analyze(df_dt)

    final_row = df_diag.iloc[-1]
    assert final_row["fault_type"] == "LUBRICATION_FAULT"
    assert final_row["severity"] in ["HIGH", "CRITICAL"]

    contribs = final_row["contributing_signals"]
    assert "oil_pressure" in contribs
    assert "oil_temperature" in contribs


def test_vibration_anomaly_classification():
    """Test 4: Verify vibration anomaly classifies as VIBRATION_ANOMALY with vibration contributor."""
    df_raw = generate_telemetry(scenario="vibration_anomaly", duration_minutes=20.0, seed=42)
    twin = DigitalTwin()
    df_dt = twin.predict_expected_state(df_raw)

    analyzer = ResidualAnalyzer()
    df_diag = analyzer.analyze(df_dt)

    final_row = df_diag.iloc[-1]
    assert final_row["fault_type"] == "VIBRATION_ANOMALY"
    assert "vibration" in final_row["contributing_signals"]


def test_sensor_drift_classification():
    """Test 5: Verify sensor drift classifies as SENSOR_DRIFT and NOT OVERHEATING."""
    df_raw = generate_telemetry(scenario="sensor_drift", drift_sensor="CHT", duration_minutes=20.0, seed=42)
    twin = DigitalTwin()
    df_dt = twin.predict_expected_state(df_raw)

    analyzer = ResidualAnalyzer()
    df_diag = analyzer.analyze(df_dt)

    final_row = df_diag.iloc[-1]
    assert final_row["fault_type"] == "SENSOR_DRIFT"
    assert final_row["fault_type"] != "OVERHEATING"
    assert final_row["contributing_signals"] == "CHT"


def test_unknown_anomaly_classification():
    """Test 6: Verify unmatched abnormal residual pattern classifies as UNKNOWN_ANOMALY."""
    analyzer = ResidualAnalyzer()

    # Synthetic residuals: fuel_flow +5.2σ while CHT/EGT/oil are nominal
    norm_res = {
        "RPM": 0.0,
        "CHT": 0.2,
        "EGT": -0.1,
        "oil_pressure": 0.0,
        "oil_temperature": 0.1,
        "fuel_flow": 5.2,  # Isolated unmodeled fuel flow spike
        "vibration": 0.0,
        "battery_voltage": 0.0,
    }

    res = analyzer.analyze_point(norm_res)
    assert res["fault_type"] == "UNKNOWN_ANOMALY"
    assert res["anomaly_score"] > 10.0



def test_temporal_persistence_logic():
    """Test 7: Verify single noisy sample does not trigger persistent anomaly, but 3 consecutive samples do."""
    analyzer = ResidualAnalyzer(persistence_window=3)

    norm_res_normal = {sig: 0.0 for sig in config.MODELED_SIGNALS}
    norm_res_spike = {sig: 0.0 for sig in config.MODELED_SIGNALS}
    norm_res_spike["CHT"] = 4.0  # > 3.0 anomaly threshold

    # Sample 1 (Spike 1)
    res1 = analyzer.analyze_point(norm_res_spike)
    assert res1["instant_flags"]["anomaly_CHT"] == 1
    assert res1["persistent_flags"]["persistent_anomaly_CHT"] == 0

    # Sample 2 (Spike 2)
    res2 = analyzer.analyze_point(norm_res_spike)
    assert res2["instant_flags"]["anomaly_CHT"] == 1
    assert res2["persistent_flags"]["persistent_anomaly_CHT"] == 0

    # Sample 3 (Spike 3 -> persistence window reached!)
    res3 = analyzer.analyze_point(norm_res_spike)
    assert res3["instant_flags"]["anomaly_CHT"] == 1
    assert res3["persistent_flags"]["persistent_anomaly_CHT"] == 1

    # Sample 4 (Return to normal)
    res4 = analyzer.analyze_point(norm_res_normal)
    assert res4["instant_flags"]["anomaly_CHT"] == 0
    assert res4["persistent_flags"]["persistent_anomaly_CHT"] == 0


def test_determinism():
    """Test 8: Verify identical input produces identical analysis result."""
    df_raw = generate_telemetry(scenario="overheating", duration_minutes=10.0, seed=42)
    twin = DigitalTwin()
    df_dt = twin.predict_expected_state(df_raw)

    analyzer1 = ResidualAnalyzer()
    df_diag1 = analyzer1.analyze(df_dt)

    analyzer2 = ResidualAnalyzer()
    df_diag2 = analyzer2.analyze(df_dt)

    pd.testing.assert_frame_equal(df_diag1, df_diag2)
