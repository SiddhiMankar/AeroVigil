# AeroVigil DT — AI-Enabled Real-Time Digital Twin System (Phases 1, 2, 3 & 4)

**AeroVigil DT** is an AI-enabled real-time Digital Twin architecture for health monitoring, fault prediction, and mission reliability enhancement of Aero-Piston Engines used in Medium-Altitude Long-Endurance (MALE) Unmanned Aerial Vehicles (UAVs).

---

## Complete System Pipeline

```text
  Synthetic Aero-Engine Telemetry
(Simulator: RPM, CHT, EGT, Oil Press/Temp, Vib, Batt)
                 │
                 ▼
  Operating Conditions (Throttle, Ambient Temp, Phase)
                 │
                 ▼
     Physics-Based Digital Twin
    (Expected-State Surrogate Model)
                 │
                 ▼
       Residual Generation Engine
    (Residual = Observed - Expected)
                 │
                 ▼
     Residual Analysis & Anomaly Detection
  (Normalized Residuals, Persistence, Scoring)
                 │
                 ▼
      Rule-Based Fault Classification
 (NORMAL, OVERHEATING, LUBRICATION, VIBRATION, SENSOR_DRIFT)
                 │
                 ▼
        Engine Health Index (0-100)
    (Smooth Weighted Health & Trend)
                 │
                 ▼
      Prototype RUL Estimation
   (Linear Slope Projection & Status)
                 │
                 ▼
       Mission Risk Assessment
 (Flight-Phase Aware Risk Score & Advisory)
                 │
                 ▼
  [Phase 5: Real-Time Dashboard & Replay]
```

---

## 1. Prototype Scope

* **Phase 1 — Repository Foundation & Synthetic Telemetry Simulator**:
  Establishes a clean, reproducible repository foundation and implements a physics-correlated synthetic engine telemetry generator supporting 8 flight mission phases and 5 controlled fault scenarios.

* **Phase 2 — Physics-Based Expected-State Model / Digital Twin**:
  Implements an independent reduced-order physics-informed surrogate estimator (`src/digital_twin.py`) that predicts expected engine states (`expected_*`) strictly from healthy operating conditions without inspecting scenario labels.

* **Phase 3 — Residual Analysis, Anomaly Detection & Fault Classification**:
  Implements an explainable diagnostic engine (`src/residual_analysis.py`). Evaluates normalized residuals ($\mathbf{z}$), applies sliding temporal persistence, calculates global anomaly scores ($0 - 100$), computes diagnostic severity levels, classifies fault types (`NORMAL`, `OVERHEATING`, `LUBRICATION_FAULT`, `VIBRATION_ANOMALY`, `SENSOR_DRIFT`, `UNKNOWN_ANOMALY`), isolates sensor drift from physical thermal failures, and transparently provides evidence scores ($0.0 - 1.0$) and contributing signal explanations.

* **Phase 4 — Engine Health Index, Prototype RUL Estimation & Mission Risk**:
  Implements the operational decision layer:
  1. **Engine Health Index** (`src/health_index.py`): Continuous $0 - 100$ health index, health state categorization (`HEALTHY`, `DEGRADED`, `WARNING`, `SEVERE`, `CRITICAL`), and trend tracking.
  2. **Prototype RUL Estimator** (`src/rul_estimator.py`): Projects remaining useful time in seconds and minutes based on recent health degradation slope ($\frac{dH}{dt}$).
  3. **Mission Risk Model** (`src/mission_risk.py`): Flight-phase aware risk score ($0 - 100$) and prototype operational advisories (`CONTINUE_MONITORING`, `INCREASE_MONITORING`, `INSPECT_AT_NEXT_OPPORTUNITY`, `CONSIDER_MISSION_ABORT`).

---

## 2. Synthetic Data & Prototype Disclaimer

> **IMPORTANT DISCLAIMER**:
> All telemetry data produced by this simulator is **synthetic prototype data**. The signals, ranges, equations, Health Index thresholds, RUL projections, and Mission Risk advisories are mathematically modeled for technical feasibility demonstration (SIH 2026 submission) and are **NOT** claimed to represent certified physical aircraft engine measurements, real-world engine failure datasets, or flight-certified safety decisions.

---

## 3. Telemetry Signal Definitions & Units

The system models 13 time-series signals sampled at configurable rates (default 1 Hz):

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

## 4. Mission Profile & Injected Fault Scenarios

### Mission Phases
1. **STARTUP** (0% – 5%), 2. **TAKEOFF** (5% – 10%), 3. **CLIMB** (10% – 25%), 4. **CRUISE** (25% – 55%), 5. **MANEUVER** (55% – 70%), 6. **CRUISE** (70% – 85%), 7. **DESCENT** (85% – 95%), 8. **LANDING** (95% – 100%).

### Fault Scenarios
1. **`normal`**: Healthy baseline mission telemetry.
2. **`overheating`**: Cooling airflow blockage / lean fuel mixture (CHT $+65^\circ\text{C}$, EGT $+85^\circ\text{C}$, Oil Temp $+28^\circ\text{C}$).
3. **`lubrication_fault`**: Oil pump degradation / leak (Oil Press $-34\text{ psi}$ drop, Oil Temp $+36^\circ\text{C}$ rise).
4. **`vibration_anomaly`**: Cylinder misfire / propeller unbalance ($+0.70\text{ g}$ vibration rise & RPM jitter).
5. **`sensor_drift`**: Isolated linear bias ($+65^\circ\text{C}$) strictly on CHT sensor probe while correlated physics remain nominal.

---

## 5. Project Structure

```
aerovigil-dt/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py                      # Telemetry schema, bounds, thresholds, signal weights, risk parameters
├── src/
│   ├── __init__.py                # Package exports
│   ├── simulator.py               # Synthetic Telemetry Simulator
│   ├── digital_twin.py            # Physics-Informed Digital Twin & Residual Engine
│   ├── residual_analysis.py       # Residual Analyzer, Anomaly Detector & Classifier
│   ├── health_index.py            # Engine Health Index & State Categorizer
│   ├── rul_estimator.py           # Prototype RUL Estimator
│   ├── mission_risk.py            # Phase-Aware Mission Risk & Advisory Model
│   └── health_risk_pipeline.py    # Integrated End-to-End AeroVigil Pipeline
├── data/
│   ├── README.md                  # Dataset specifications & disclaimers
│   ├── normal.csv                 # 20-min normal mission telemetry (1200 rows)
│   ├── overheating.csv            # Thermal degradation telemetry (1200 rows)
│   ├── lubrication_fault.csv      # Lubrication pressure loss telemetry (1200 rows)
│   ├── vibration_fault.csv        # Mechanical vibration & jitter telemetry (1200 rows)
│   └── sensor_drift.csv           # Isolated sensor bias telemetry (1200 rows)
├── scripts/
│   ├── generate_datasets.py       # Script to generate CSV datasets in data/
│   ├── validate_visuals.py        # Simulator visual validation script
│   ├── validate_digital_twin.py   # Digital Twin validation & plotting script
│   ├── validate_fault_detection.py# Fault detection, confusion matrix & diagnostic plotting
│   └── validate_health_rul_risk.py# Health Index, RUL & Mission Risk validation & plotting
├── tests/
│   ├── test_simulator.py          # Phase 1 simulator test suite (10 tests)
│   ├── test_digital_twin.py       # Phase 2 Digital Twin test suite (9 tests)
│   ├── test_residual_analysis.py  # Phase 3 Residual Analysis test suite (8 tests)
│   └── test_health_rul_risk.py    # Phase 4 Health Index, RUL & Risk test suite (10 tests)
└── docs/
    ├── telemetry_spec.md          # Telemetry signal & physics specification
    ├── digital_twin_model.md      # Digital Twin model equations & residual specification
    ├── residual_analysis.md       # Residual analysis, rule signatures & severity specification
    ├── health_index.md            # Health Index formulation & state specification
    ├── rul_estimation.md          # Prototype RUL estimation specification
    ├── mission_risk.md            # Mission risk & flight phase sensitivity specification
    └── plots/
        ├── telemetry_comparison.png         # Simulator scenario visual plots
        ├── digital_twin_validation.png      # Digital Twin expected vs actual residual plots
        ├── residual_analysis_validation.png # Anomaly score, residual signature & fault timeline plots
        ├── health_index_comparison.png      # Health Index trajectories across scenarios
        ├── rul_trajectory.png               # Prototype RUL trajectories under degradation
        ├── mission_risk_timeline.png        # Mission risk score timelines across scenarios
        └── integrated_engine_state.png      # 4-Panel end-to-end engine state overview
```

---

## 6. Running the Pipeline & Generating Data

### Install Dependencies
```bash
pip install -r requirements.txt
```

### End-to-End Python API Example
```python
from src.simulator import generate_telemetry
from src.health_risk_pipeline import run_pipeline

# 1. Generate raw telemetry
df_raw = generate_telemetry(scenario="overheating", duration_minutes=20.0, seed=42)

# 2. Run full AeroVigil DT pipeline
df_out = run_pipeline(df_raw)

# Inspect final operational decision outputs
final = df_out.iloc[-1]
print(f"Fault Type    : {final['fault_type']}")
print(f"Health Index  : {final['health_index']} ({final['health_state']})")
print(f"RUL Estimate  : {final['rul_minutes']} min ({final['rul_status']})")
print(f"Mission Risk  : {final['mission_risk_score']} ({final['mission_risk_level']})")
print(f"Advisory      : {final['mission_recommendation']}")
```

### Run Integrated Health, RUL & Risk Validation
To execute the full pipeline evaluation across all 5 scenarios and generate plots:
```bash
python scripts/validate_health_rul_risk.py
```

---

## 7. Running Automated Tests

Run the full pytest suite (37 test cases across Phases 1, 2, 3, and 4):

```bash
python -m pytest tests/ -v
```
