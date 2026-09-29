# AeroVigil DT — AI-Enabled Real-Time Digital Twin System (Phases 1–5)

**AeroVigil DT** is an AI-enabled real-time Digital Twin architecture for health monitoring, fault prediction, and mission reliability enhancement of Aero-Piston Engines used in Medium-Altitude Long-Endurance (MALE) Unmanned Aerial Vehicles (UAVs).

---

## End-to-End Architecture

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
   Streamlit Real-Time Dashboard & Replay
         (app.py & Mission Controls)
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
  Implements the operational decision layer (`src/health_index.py`, `src/rul_estimator.py`, `src/mission_risk.py`, `src/health_risk_pipeline.py`) providing continuous Health Index ($0 - 100$), degradation-slope RUL estimation, flight-phase aware mission risk scoring ($0 - 100$), and prototype operational advisories (`CONTINUE_MONITORING`, `INCREASE_MONITORING`, `INSPECT_AT_NEXT_OPPORTUNITY`, `CONSIDER_MISSION_ABORT`).

* **Phase 5 — Real-Time Digital Twin Dashboard & Mission Replay**:
  Implements an interactive web application (`app.py`) built with Streamlit. Exposes the complete analytical pipeline, KPI metrics, live telemetry cards, Digital Twin observed vs expected charts, residual signatures, health trajectories, RUL countdowns, mission risk timelines, explainable fault diagnostics, mission replay slider, and cross-scenario evaluation comparison matrices.

---

## 2. Synthetic Data & Prototype Disclaimer

> **IMPORTANT DISCLAIMER**:
> All telemetry data produced by this simulator is **synthetic prototype data**. The signals, ranges, equations, Health Index thresholds, RUL projections, and Mission Risk advisories are mathematically modeled for technical feasibility demonstration (SIH 2026 submission) and are **NOT** claimed to represent certified physical aircraft engine measurements, real-world engine failure datasets, or flight-certified safety decisions.

---

## 3. Running the Streamlit Dashboard

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Launch Dashboard
```bash
streamlit run app.py
```

Upon launching, the dashboard defaults to **Overheating Thermal Fault** (Seed 42). Use the sidebar to switch scenarios, adjust mission duration, step through the **Mission Replay Slider**, or inspect the **Cross-Scenario Evaluation Matrix**.

---

## 4. Telemetry Signal Definitions & Units

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

## 5. Project Structure

```
aerovigil-dt/
├── app.py                         # Streamlit Real-Time Dashboard & Mission Replay App
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
│   ├── health_risk_pipeline.py    # Integrated End-to-End AeroVigil Pipeline
│   └── dashboard_utils.py         # Dashboard Helper Utilities & Cached Processing
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
│   ├── test_health_rul_risk.py    # Phase 4 Health Index, RUL & Risk test suite (10 tests)
│   └── test_dashboard_pipeline.py # Phase 5 Dashboard integration test suite (10 tests)
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

## 6. End-to-End Python API Example

```python
from src.simulator import generate_telemetry
from src.health_risk_pipeline import run_pipeline

# 1. Generate telemetry (e.g., overheating scenario)
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

---

## 7. Running Automated Tests

Run the full pytest suite (47 test cases across Phases 1–5):

```bash
python -m pytest tests/ -v
```
