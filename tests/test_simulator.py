"""
AeroVigil DT - Simulator Automated Test Suite
Verifies simulator execution, schema integrity, physical bounds, fault injection impacts,
reproducibility, and distinction between physical degradation vs sensor drift.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure root workspace is in import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator import generate_telemetry, EngineSimulator
import config


def test_simulator_execution():
    """Test 1: Verify simulator runs and produces non-empty DataFrame of expected shape."""
    df = generate_telemetry(scenario="normal", duration_minutes=5.0, sample_rate_hz=1.0, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 300  # 5 mins * 60s
    assert not df.empty


def test_required_columns():
    """Test 2: Verify all 13 required telemetry columns are present."""
    df = generate_telemetry(scenario="normal", seed=42)
    for col in config.REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"


def test_no_nan_or_inf():
    """Test 3 & 4: Ensure zero NaN, Null, or Infinite values exist in output."""
    for scenario in config.FAULT_SCENARIOS:
        df = generate_telemetry(scenario=scenario, seed=42)
        assert not df.isna().any().any(), f"NaN value found in scenario: {scenario}"
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert not np.isinf(df[numeric_cols].values).any(), f"Inf value found in scenario: {scenario}"


def test_numerical_bounds():
    """Test 5 & 6: Verify signals stay within defined physical prototype bounds."""
    df = generate_telemetry(scenario="normal", duration_minutes=20.0, seed=42)

    # Throttle in [0, 1]
    assert df["throttle"].min() >= config.PHYSICAL_BOUNDS["throttle"][0]
    assert df["throttle"].max() <= config.PHYSICAL_BOUNDS["throttle"][1]

    # RPM within prototype range [0, 6500]
    assert df["RPM"].min() >= config.PHYSICAL_BOUNDS["RPM"][0]
    assert df["RPM"].max() <= config.PHYSICAL_BOUNDS["RPM"][1]

    # CHT & EGT positive and reasonable
    assert df["CHT"].min() >= 15.0
    assert df["CHT"].max() <= config.PHYSICAL_BOUNDS["CHT"][1]
    assert df["EGT"].min() >= 15.0
    assert df["EGT"].max() <= config.PHYSICAL_BOUNDS["EGT"][1]

    # Battery Voltage in [9, 15]
    assert df["battery_voltage"].min() >= config.PHYSICAL_BOUNDS["battery_voltage"][0]
    assert df["battery_voltage"].max() <= config.PHYSICAL_BOUNDS["battery_voltage"][1]


def test_fault_scenario_effects():
    """Test 7 & 9: Verify fault scenarios modify target signals relative to normal baseline."""
    df_norm = generate_telemetry(scenario="normal", seed=42)
    df_over = generate_telemetry(scenario="overheating", seed=42)
    df_lube = generate_telemetry(scenario="lubrication_fault", seed=42)
    df_vib = generate_telemetry(scenario="vibration_anomaly", seed=42)

    # Overheating increases mean CHT and EGT in second half of mission
    half_idx = len(df_norm) // 2
    assert df_over["CHT"].iloc[half_idx:].mean() > df_norm["CHT"].iloc[half_idx:].mean() + 20.0
    assert df_over["EGT"].iloc[half_idx:].mean() > df_norm["EGT"].iloc[half_idx:].mean() + 30.0

    # Lubrication fault significantly reduces oil pressure & increases oil temp
    assert df_lube["oil_pressure"].iloc[half_idx:].mean() < df_norm["oil_pressure"].iloc[half_idx:].mean() - 15.0
    assert df_lube["oil_temperature"].iloc[half_idx:].mean() > df_norm["oil_temperature"].iloc[half_idx:].mean() + 10.0

    # Vibration anomaly increases mean vibration g-level
    assert df_vib["vibration"].iloc[half_idx:].mean() > df_norm["vibration"].iloc[half_idx:].mean() + 0.20


test_reproducibility_data = [
    ("normal", 42),
    ("overheating", 123),
    ("sensor_drift", 999),
]

@pytest.mark.parametrize("scenario, seed", test_reproducibility_data)
def test_reproducibility(scenario, seed):
    """Test 8: Verify deterministic output when using identical random seed."""
    df1 = generate_telemetry(scenario=scenario, seed=seed)
    df2 = generate_telemetry(scenario=scenario, seed=seed)
    pd.testing.assert_frame_equal(df1, df2)

    # Different seed produces different results
    df3 = generate_telemetry(scenario=scenario, seed=seed + 1)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df1, df3)


def test_sensor_drift_isolation():
    """Verify sensor drift modifies ONLY the target sensor (e.g. CHT) while correlated physics match normal baseline."""
    df_norm = generate_telemetry(scenario="normal", seed=42)
    df_drift = generate_telemetry(scenario="sensor_drift", drift_sensor="CHT", seed=42)

    half_idx = len(df_norm) // 2

    # CHT is significantly shifted upward in drift scenario
    assert df_drift["CHT"].iloc[half_idx:].mean() > df_norm["CHT"].iloc[half_idx:].mean() + 20.0

    # But underlying physical EGT and Oil Temp remain nearly identical to normal baseline!
    egt_diff = np.abs(df_drift["EGT"] - df_norm["EGT"]).max()
    oil_temp_diff = np.abs(df_drift["oil_temperature"] - df_norm["oil_temperature"]).max()

    assert egt_diff < 1e-3, f"EGT unexpectedly changed in CHT sensor drift! max diff: {egt_diff}"
    assert oil_temp_diff < 1e-3, f"Oil temperature unexpectedly changed in CHT sensor drift! max diff: {oil_temp_diff}"


def test_mission_phases_represented():
    """Verify all 8 mission phases are represented in output."""
    df = generate_telemetry(scenario="normal", seed=42)
    phases_in_data = set(df["mission_phase"].unique())
    expected_phases = {p["name"] for p in config.MISSION_PHASES}
    assert phases_in_data == expected_phases
