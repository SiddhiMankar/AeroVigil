"""
AeroVigil DT - Dashboard Utility & Pipeline Integration Test Suite
Phase 5 Component

Verifies scenario mappings, cached simulation execution, snapshot extraction,
NaN/UNAVAILABLE RUL handling, and scenario comparison matrix generation.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure repo root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dashboard_utils import (
    SCENARIO_DISPLAY_NAMES,
    SCENARIO_NAME_MAP,
    extract_snapshot_at_index,
    generate_scenario_comparison,
    run_cached_simulation,
)
import config


def test_scenario_mapping_consistency():
    """Test 1: Verify display names map bijectively to simulator scenario keys."""
    for key, display in SCENARIO_DISPLAY_NAMES.items():
        assert key in config.FAULT_SCENARIOS
        assert SCENARIO_NAME_MAP[display] == key


@pytest.mark.parametrize("scenario", config.FAULT_SCENARIOS)
def test_run_cached_simulation_scenarios(scenario):
    """Test 2: Verify run_cached_simulation executes across all 5 scenarios."""
    df = run_cached_simulation(scenario=scenario, duration_minutes=5.0, seed=42)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 300
    assert "health_index" in df.columns
    assert "mission_risk_score" in df.columns
    assert "fault_type" in df.columns


def test_extract_snapshot_at_index_normal():
    """Test 3: Verify extract_snapshot_at_index handles normal healthy state and NaN RUL."""
    df = run_cached_simulation(scenario="normal", duration_minutes=5.0, seed=42)
    snapshot = extract_snapshot_at_index(df, idx=150)

    assert isinstance(snapshot, dict)
    assert snapshot["health_index"] >= 90.0
    assert snapshot["fault_type"] == "NORMAL"
    assert snapshot["rul_display"] == "--"
    assert snapshot["rul_status"] in ["STABLE", "UNAVAILABLE"]
    assert snapshot["mission_risk_level"] == "LOW"


def test_extract_snapshot_at_index_degradation():
    """Test 4: Verify snapshot extraction during active fault degradation."""
    df = run_cached_simulation(scenario="overheating", duration_minutes=20.0, seed=42)
    snapshot = extract_snapshot_at_index(df, idx=len(df) - 1)

    assert snapshot["fault_type"] == "OVERHEATING"
    assert snapshot["severity"] in ["HIGH", "CRITICAL"]
    assert snapshot["mission_risk_level"] in ["HIGH", "CRITICAL"]
    assert snapshot["health_index"] < 30.0


def test_generate_scenario_comparison_matrix():
    """Test 5: Verify scenario comparison matrix generates 5 rows of summary metrics."""
    comp_df = generate_scenario_comparison(duration_minutes=5.0, seed=42)

    assert isinstance(comp_df, pd.DataFrame)
    assert len(comp_df) == 5
    assert "Scenario" in comp_df.columns
    assert "Final Health" in comp_df.columns
    assert "Final Risk Level" in comp_df.columns
    assert "Detected Fault" in comp_df.columns


def test_deterministic_seed_reproducibility():
    """Test 6: Verify identical random seed produces bitwise identical dashboard snapshots."""
    df1 = run_cached_simulation(scenario="overheating", duration_minutes=5.0, seed=123)
    df2 = run_cached_simulation(scenario="overheating", duration_minutes=5.0, seed=123)

    pd.testing.assert_frame_equal(df1, df2)

    snap1 = extract_snapshot_at_index(df1, idx=200)
    snap2 = extract_snapshot_at_index(df2, idx=200)

    assert snap1["health_index"] == snap2["health_index"]
    assert snap1["anomaly_score"] == snap2["anomaly_score"]
