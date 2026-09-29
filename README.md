# AeroVigil DT — AI-Enabled Real-Time Digital Twin System (Phase 1)

**AeroVigil DT** is an AI-enabled real-time Digital Twin architecture for health monitoring, fault prediction, and mission reliability enhancement of Aero-Piston Engines used in Medium-Altitude Long-Endurance (MALE) Unmanned Aerial Vehicles (UAVs).

---

## 1. Prototype Scope (Phase 1)

This repository contains **Phase 1: Repository Foundation + Synthetic Aero-Piston Engine Telemetry Simulator**.

Phase 1 establishes a clean, reproducible repository foundation and implements a physics-correlated synthetic telemetry generator. This generator models flight mission profiles and injects controlled thermal, mechanical, lubrication, and sensor fault scenarios to produce time-series datasets consumed by downstream Digital Twin phases (Physics-based Expected State Model, Residual Analysis, Anomaly Detection, Fault Classification, RUL Estimation, and Mission Risk Assessment).

---

## 2. Synthetic Data Disclaimer

> **IMPORTANT DISCLAIMER**:
> All telemetry data produced by this simulator is **synthetic prototype data**. The signals, ranges, equations, and values are mathematically modeled for technical feasibility demonstration (SIH 2026 submission) and are **NOT** claimed to represent certified physical aircraft engine measurements or specific published UAV engine specifications.

---

## 3. Telemetry Signal Definitions & Units

The simulator generates 13 time-series signals sampled at configurable rates (default 1 Hz):

| Signal | Unit | Description |
|---|---|---|
| `timestamp` | ISO-8601 UTC | Timestamp of telemetry reading |
| `timestamp_sec` | seconds | Elapsed seconds from mission start |
| `mission_phase` | Categorical Enum | Active flight profile phase |
| `throttle` | ratio (0.0 - 1.0) | Engine throttle command input |
| `ambient_temperature` | °C | Outside air temperature |
| `RPM` | rev/min | Engine crankshaft rotation speed |
| `CHT` | °C | Cylinder Head Temperature |
| `EGT` | °C | Exhaust Gas Temperature |
| `oil_pressure` | psi | Engine lubrication oil pressure |
| `oil_temperature` | °C | Engine lubrication oil temperature |
| `fuel_flow` | L/h | Fuel consumption rate in Liters per hour |
| `vibration` | g (normalized) | Engine block vibration amplitude |
| `battery_voltage` | V | Electrical system bus voltage |

---

## 4. Mission Profile

The flight profile models 8 continuous mission phases with smooth thermodynamic and operator load transitions:

1. **STARTUP** (0% – 5%): Engine crank, idle spool-up (~1200 RPM), alternator bus charging (~13.8V).
2. **TAKEOFF** (5% – 10%): Maximum throttle (1.00), peak RPM (~5600 RPM), high fuel flow (~42 L/h).
3. **CLIMB** (10% – 25%): High power climb (0.85 throttle), altitude temperature lapse.
4. **CRUISE** (25% – 55%): Steady fuel economy cruise (0.65 throttle, ~4600 RPM).
5. **MANEUVER** (55% – 70%): Dynamic throttle variations (0.75 avg) simulating tactical maneuvers.
6. **CRUISE** (70% – 85%): Return to steady cruise altitude.
7. **DESCENT** (85% – 95%): Reduced throttle (0.30, ~3000 RPM), thermal cooling.
8. **LANDING** (95% – 100%): Idle touchdown (0.15 throttle, ~1500 RPM) and shutdown.

---

## 5. Injected Fault Scenarios

The simulator supports 5 reproducible scenario configurations (starting progressively at 30% mission duration):

1. **`normal`**: Healthy baseline mission telemetry.
2. **`overheating`**: Simulated cooling air duct blockage / lean mixture leading to gradual increases in CHT (+65°C), EGT (+85°C), and Oil Temperature (+28°C).
3. **`lubrication_fault`**: Simulated oil pump degradation / oil leak causing oil pressure loss (-34 psi down to ~15 psi) and excessive oil temperature rise (+36°C).
4. **`vibration_anomaly`**: Cylinder misfire or propeller imbalance causing elevated vibration (+0.70g) and RPM rotational jitter.
5. **`sensor_drift`**: Introduces a gradual linear bias (+65°C) strictly into a single targeted sensor (e.g., CHT) while underlying physical engine parameters (EGT, Oil Temp, RPM, Oil Press) remain **completely normal**. This enables downstream residual analysis to isolate sensor faults from actual engine failures.

---

## 6. Project Structure

```
aerovigil-dt/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py
├── src/
│   ├── __init__.py
│   └── simulator.py
├── data/
│   ├── README.md
│   ├── normal.csv
│   ├── overheating.csv
│   ├── lubrication_fault.csv
│   ├── vibration_fault.csv
│   └── sensor_drift.csv
├── scripts/
│   ├── generate_datasets.py
│   └── validate_visuals.py
├── tests/
│   └── test_simulator.py
└── docs/
    ├── telemetry_spec.md
    └── plots/
        └── telemetry_comparison.png
```

---

## 7. Running the Simulator & Generating Data

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Python API Example
```python
from src.simulator import generate_telemetry

# Generate overheating scenario with seed 42
df = generate_telemetry(scenario="overheating", duration_minutes=20.0, seed=42)
print(df.head())
```

### Generate CSV Datasets
To regenerate all 5 CSV datasets into `data/`:
```bash
python scripts/generate_datasets.py
```

### Generate Visual Validation Plot
To produce signal comparison plots in `docs/plots/`:
```bash
python scripts/validate_visuals.py
```

---

## 8. Running Automated Tests

Run the full pytest suite from the project root:

```bash
python -m pytest tests/ -v
```
