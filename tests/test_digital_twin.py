"""
AeroVigil DT - Digital Twin Automated Test Suite
Verifies DigitalTwin instantiation, schema completeness, determinism, state access,
residual generation, and physics-informed fault scenario discrimination.
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
import config


def test_digital_twin_instantiation_and_state():
    """Test 1: Verify DigitalTwin can be instantiated, reset, and state fetched."""
    twin = DigitalTwin(sample_rate_hz=1.0)
    state = twin.get_state()
    assert isinstance(state, dict)
    assert "step_counter" in state
    assert state["step_counter"] == 0
    assert "curr_rpm" in state


def test_expected_state_schema():
    """Test 2: Verify expected state and residual columns are appended cleanly."""
    df_raw = generate_telemetry(scenario="normal", duration_minutes=5.0, seed=42)
    twin = DigitalTwin()
    res_df = twin.predict_expected_state(df_raw)

    # Check original columns preserved
    for col in config.REQUIRED_COLUMNS:
        assert col in res_df.columns

    # Check expected, residual, and normalized residual columns exist
    for sig in config.MODELED_SIGNALS:
        assert f"expected_{sig}" in res_df.columns, f"Missing expected column for {sig}"
        assert f"residual_{sig}" in res_df.columns, f"Missing residual column for {sig}"
        assert f"normalized_residual_{sig}" in res_df.columns, f"Missing normalized residual column for {sig}"

    # Check no NaN values introduced
    assert not res_df.isna().any().any()


def test_determinism():
    """Test 3: Verify identical telemetry input yields identical expected output."""
    df_raw = generate_telemetry(scenario="normal", duration_minutes=10.0, seed=42)

    twin1 = DigitalTwin()
    res_df1 = twin1.predict_expected_state(df_raw)

    twin2 = DigitalTwin()
    res_df2 = twin2.predict_expected_state(df_raw)

    pd.testing.assert_frame_equal(res_df1, res_df2)


def test_normal_telemetry_bounded_residuals():
    """Test 4: Verify expected residuals under healthy normal operation are small and bounded."""
    df_raw = generate_telemetry(scenario="normal", duration_minutes=20.0, seed=42)
    twin = DigitalTwin()
    res_df = twin.predict_expected_state(df_raw)

    # Residual mean across mission should be close to 0
    assert abs(res_df["residual_CHT"].mean()) < 3.0
    assert abs(res_df["residual_EGT"].mean()) < 5.0
    assert abs(res_df["residual_oil_pressure"].mean()) < 2.0
    assert abs(res_df["residual_oil_temperature"].mean()) < 2.0
    assert abs(res_df["residual_vibration"].mean()) < 0.05

    # Normalized residuals under normal should stay mostly within [-3, 3] std
    second_half = res_df.iloc[len(res_df) // 2 :]
    assert abs(second_half["normalized_residual_CHT"].mean()) < 1.5
    assert abs(second_half["normalized_residual_EGT"].mean()) < 1.5


def test_overheating_residuals():
    """Test 5: Verify overheating scenario generates strong positive CHT, EGT, and Oil Temp residuals."""
    df_raw = generate_telemetry(scenario="overheating", duration_minutes=20.0, seed=42)
    twin = DigitalTwin()
    res_df = twin.predict_expected_state(df_raw)

    second_half = res_df.iloc[len(res_df) // 2 :]

    # CHT, EGT, Oil Temp residuals should be significantly positive
    assert second_half["residual_CHT"].mean() > 20.0
    assert second_half["residual_EGT"].mean() > 30.0
    assert second_half["residual_oil_temperature"].mean() > 10.0

    # Normalized CHT residual should exceed 3-sigma anomaly threshold
    assert second_half["normalized_residual_CHT"].mean() > 5.0


def test_lubrication_fault_residuals():
    """Test 6: Verify lubrication fault generates negative oil pressure residual and positive oil temp residual."""
    df_raw = generate_telemetry(scenario="lubrication_fault", duration_minutes=20.0, seed=42)
    twin = DigitalTwin()
    res_df = twin.predict_expected_state(df_raw)

    second_half = res_df.iloc[len(res_df) // 2 :]

    # Oil pressure residual significantly negative (observed < expected)
    assert second_half["residual_oil_pressure"].mean() < -10.0
    assert second_half["normalized_residual_oil_pressure"].mean() < -5.0

    # Oil temp residual positive (friction heat)
    assert second_half["residual_oil_temperature"].mean() > 10.0


def test_vibration_anomaly_residuals():
    """Test 7: Verify vibration anomaly generates strong positive vibration residual."""
    df_raw = generate_telemetry(scenario="vibration_anomaly", duration_minutes=20.0, seed=42)
    twin = DigitalTwin()
    res_df = twin.predict_expected_state(df_raw)

    second_half = res_df.iloc[len(res_df) // 2 :]

    assert second_half["residual_vibration"].mean() > 0.20
    assert second_half["normalized_residual_vibration"].mean() > 5.0


def test_sensor_drift_isolation_residuals():
    """Test 8: Verify sensor drift produces strong CHT residual while correlated physical signals remain nominal."""
    df_raw = generate_telemetry(scenario="sensor_drift", drift_sensor="CHT", duration_minutes=20.0, seed=42)
    twin = DigitalTwin()
    res_df = twin.predict_expected_state(df_raw)

    second_half = res_df.iloc[len(res_df) // 2 :]

    # Targeted CHT sensor shows huge residual anomaly
    assert second_half["residual_CHT"].mean() > 25.0
    assert second_half["normalized_residual_CHT"].mean() > 8.0

    # Non-drifted physical signals (EGT, Oil Temp, Oil Pressure) remain near zero baseline!
    assert abs(second_half["residual_EGT"].mean()) < 2.0
    assert abs(second_half["residual_oil_temperature"].mean()) < 2.0
    assert abs(second_half["residual_oil_pressure"].mean()) < 1.0


def test_online_predict_step():
    """Test 9: Verify online single-step predict_step API works seamlessly."""
    twin = DigitalTwin()

    step1 = twin.predict_step(throttle=0.8, ambient_temperature=15.0, mission_phase="CLIMB")
    assert "expected_RPM" in step1
    assert "expected_CHT" in step1
    assert step1["expected_RPM"] > 1000.0

    # Test with observed telemetry dict
    obs = {"RPM": 5100.0, "CHT": 140.0, "EGT": 680.0, "oil_pressure": 55.0}
    step2 = twin.predict_step(throttle=0.8, ambient_temperature=15.0, mission_phase="CLIMB", observed_telemetry=obs)

    assert "residual_RPM" in step2
    assert "residual_CHT" in step2
    assert "normalized_residual_CHT" in step2
