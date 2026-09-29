"""
AeroVigil DT - Configuration & Spec Parameters
Synthetic Aero-Piston Engine Telemetry Simulator Configuration
"""

from typing import Dict, List, Any

# Sensor Definitions & Units Schema
TELEMETRY_SIGNALS: Dict[str, Dict[str, str]] = {
    "timestamp": {
        "unit": "ISO-8601 UTC String",
        "description": "Timestamp of telemetry reading",
    },
    "timestamp_sec": {
        "unit": "seconds",
        "description": "Elapsed seconds from mission start",
    },
    "mission_phase": {
        "unit": "categorical enum",
        "description": "Active flight mission phase",
    },
    "throttle": {
        "unit": "ratio (0.0 - 1.0)",
        "description": "Engine throttle command",
    },
    "ambient_temperature": {
        "unit": "°C",
        "description": "Ambient outside air temperature",
    },
    "RPM": {
        "unit": "rev/min",
        "description": "Engine crankshaft rotation speed",
    },
    "CHT": {
        "unit": "°C",
        "description": "Cylinder Head Temperature",
    },
    "EGT": {
        "unit": "°C",
        "description": "Exhaust Gas Temperature",
    },
    "oil_pressure": {
        "unit": "psi",
        "description": "Engine lubrication oil pressure",
    },
    "oil_temperature": {
        "unit": "°C",
        "description": "Engine lubrication oil temperature",
    },
    "fuel_flow": {
        "unit": "L/h",
        "description": "Fuel consumption rate in Liters per hour",
    },
    "vibration": {
        "unit": "g (normalized)",
        "description": "Engine block vibration amplitude in g-forces",
    },
    "battery_voltage": {
        "unit": "V",
        "description": "Electrical system bus voltage",
    },
}

REQUIRED_COLUMNS: List[str] = [
    "timestamp",
    "timestamp_sec",
    "mission_phase",
    "throttle",
    "ambient_temperature",
    "RPM",
    "CHT",
    "EGT",
    "oil_pressure",
    "oil_temperature",
    "fuel_flow",
    "vibration",
    "battery_voltage",
]

# Mission Phase Profile Timeline Definitions (relative fractions of total duration)
MISSION_PHASES: List[Dict[str, Any]] = [
    {"name": "STARTUP", "duration_fraction": 0.05, "throttle_target": 0.10, "alt_temp_offset": 0.0},
    {"name": "TAKEOFF", "duration_fraction": 0.05, "throttle_target": 1.00, "alt_temp_offset": -0.5},
    {"name": "CLIMB", "duration_fraction": 0.15, "throttle_target": 0.85, "alt_temp_offset": -3.0},
    {"name": "CRUISE", "duration_fraction": 0.30, "throttle_target": 0.65, "alt_temp_offset": -6.0},
    {"name": "MANEUVER", "duration_fraction": 0.15, "throttle_target": 0.75, "alt_temp_offset": -5.0},
    {"name": "CRUISE", "duration_fraction": 0.15, "throttle_target": 0.65, "alt_temp_offset": -6.0},
    {"name": "DESCENT", "duration_fraction": 0.10, "throttle_target": 0.30, "alt_temp_offset": -2.0},
    {"name": "LANDING", "duration_fraction": 0.05, "throttle_target": 0.15, "alt_temp_offset": 0.0},
]

# Prototype Nominal Baseline Ranges (for validation tests)
PHYSICAL_BOUNDS = {
    "throttle": (0.0, 1.0),
    "RPM": (0.0, 6500.0),
    "CHT": (0.0, 350.0),
    "EGT": (0.0, 1000.0),
    "oil_pressure": (0.0, 120.0),
    "oil_temperature": (0.0, 180.0),
    "fuel_flow": (0.0, 80.0),
    "vibration": (0.0, 5.0),
    "battery_voltage": (8.0, 16.0),
}

FAULT_SCENARIOS = [
    "normal",
    "overheating",
    "lubrication_fault",
    "vibration_anomaly",
    "sensor_drift",
]

# Digital Twin Modeled Signals & Residual Specs
MODELED_SIGNALS: List[str] = [
    "RPM",
    "CHT",
    "EGT",
    "oil_pressure",
    "oil_temperature",
    "fuel_flow",
    "vibration",
    "battery_voltage",
]

EXPECTED_COLUMNS: List[str] = [f"expected_{sig}" for sig in MODELED_SIGNALS]
RESIDUAL_COLUMNS: List[str] = [f"residual_{sig}" for sig in MODELED_SIGNALS]
NORMALIZED_RESIDUAL_COLUMNS: List[str] = [f"normalized_residual_{sig}" for sig in MODELED_SIGNALS]

# Nominal standard deviations under healthy normal operation (for normalized residual scoring)
NOMINAL_STD_DEV: Dict[str, float] = {
    "RPM": 35.0,
    "CHT": 3.0,
    "EGT": 8.0,
    "oil_pressure": 1.5,
    "oil_temperature": 1.2,
    "fuel_flow": 0.5,
    "vibration": 0.025,
    "battery_voltage": 0.08,
}

# Phase 3 Residual Analysis & Anomaly Detection Parameters
WARNING_THRESHOLD_STD: float = 2.0
ANOMALY_THRESHOLD_STD: float = 3.0
CRITICAL_THRESHOLD_STD: float = 4.5
DEFAULT_PERSISTENCE_WINDOW: int = 3

FAULT_TYPES: List[str] = [
    "NORMAL",
    "OVERHEATING",
    "LUBRICATION_FAULT",
    "VIBRATION_ANOMALY",
    "SENSOR_DRIFT",
    "UNKNOWN_ANOMALY",
]

SEVERITY_LEVELS: List[str] = [
    "NORMAL",
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
]

SIGNAL_WEIGHTS: Dict[str, float] = {
    "CHT": 1.5,
    "EGT": 1.5,
    "oil_pressure": 1.5,
    "oil_temperature": 1.2,
    "vibration": 1.2,
    "RPM": 1.0,
    "fuel_flow": 0.8,
    "battery_voltage": 0.5,
}

# Phase 4 Health Index Configuration Parameters
HEALTH_STATE_THRESHOLDS: Dict[str, float] = {
    "HEALTHY": 90.0,
    "DEGRADED": 75.0,
    "WARNING": 50.0,
    "SEVERE": 25.0,
    "CRITICAL": 0.0,
}

HEALTH_PENALTY_WEIGHTS: Dict[str, float] = {
    "anomaly_score": 0.40,
    "severity": 0.30,
    "persistence": 0.15,
    "fault_type": 0.15,
}

SEVERITY_PENALTY_MAP: Dict[str, float] = {
    "NORMAL": 0.0,
    "LOW": 10.0,
    "MEDIUM": 25.0,
    "HIGH": 50.0,
    "CRITICAL": 85.0,
}

FAULT_TYPE_PENALTY_MAP: Dict[str, float] = {
    "NORMAL": 0.0,
    "SENSOR_DRIFT": 20.0,
    "VIBRATION_ANOMALY": 35.0,
    "OVERHEATING": 60.0,
    "LUBRICATION_FAULT": 75.0,
    "UNKNOWN_ANOMALY": 40.0,
}

HEALTH_SMOOTHING_ALPHA: float = 0.15

# Phase 4 Prototype RUL Estimator Configuration Parameters
RUL_WINDOW_SECONDS: float = 45.0
CRITICAL_HEALTH_THRESHOLD: float = 25.0
MIN_DEGRADATION_RATE: float = -0.01  # health points per second threshold to trigger RUL calculation

# Phase 4 Mission Risk Model Configuration Parameters
MISSION_PHASE_MULTIPLIERS: Dict[str, float] = {
    "STARTUP": 0.8,
    "TAKEOFF": 1.5,
    "CLIMB": 1.3,
    "CRUISE": 1.0,
    "MANEUVER": 1.2,
    "DESCENT": 1.2,
    "LANDING": 1.4,
}

RISK_LEVEL_THRESHOLDS: Dict[str, float] = {
    "LOW": 20.0,
    "MODERATE": 45.0,
    "HIGH": 70.0,
    "CRITICAL": 100.0,
}



