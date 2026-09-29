# AeroVigil DT — AI-Enabled Real-Time Digital Twin System (Phases 1, 2 & 3)

**AeroVigil DT** is an AI-enabled real-time Digital Twin architecture for health monitoring, fault prediction, and mission reliability enhancement of Aero-Piston Engines used in Medium-Altitude Long-Endurance (MALE) Unmanned Aerial Vehicles (UAVs).

---

## Pipeline Architecture

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
    [Phase 4: Health Index, RUL & Mission Risk]
```

---

## 1. Prototype Scope

* **Phase 1 — Repository Foundation & Synthetic Telemetry Simulator**:
  Establishes a clean, reproducible repository foundation and implements a physics-correlated synthetic engine telemetry generator supporting 8 flight mission phases and 5 controlled fault scenarios.

* **Phase 2 — Physics-Based Expected-State Model / Digital Twin**:
  Implements an independent reduced-order physics-informed surrogate estimator (`src/digital_twin.py`) that predicts expected engine states (`expected_*`) strictly from healthy operating conditions without inspecting scenario labels.

* **Phase 3 — Residual Analysis, Anomaly Detection & Fault Classification**:
  Implements an explainable diagnostic engine (`src/residual_analysis.py`). Evaluates normalized residuals ($\mathbf{z}$), applies sliding temporal persistence, calculates global anomaly scores ($0 - 100$), computes diagnostic severity levels, classifies fault types (`NORMAL`, `OVERHEATING`, `LUBRICATION_FAULT`, `VIBRATION_ANOMALY`, `SENSOR_DRIFT`, `UNKNOWN_ANOMALY`), isolates sensor drift from physical thermal failures, and transparently provides evidence scores ($0.0 - 1.0$) and contributing signal explanations.

---

## 2. Synthetic Data Disclaimer

> **IMPORTANT DISCLAIMER**:
> All telemetry data produced by this simulator is **synthetic prototype data**. The signals, ranges, equations, and values are mathematically modeled for technical feasibility demonstration (SIH 2026 submission) and are **NOT** claimed to represent certified physical aircraft engine measurements or specific published UAV engine specifications.

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
├── config.py                      # Telemetry schema, bounds, thresholds, signal weights
├── src/
│   ├── __init__.py                # Package exports
│   ├── simulator.py               # Synthetic Telemetry Simulator
│   ├── digital_twin.py            # Physics-Informed Digital Twin & Residual Engine
│   └── residual_analysis.py       # Residual Analyzer, Anomaly Detector & Classifier
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
│   └── validate_fault_detection.py# Fault detection, confusion matrix & diagnostic plotting
├── tests/
│   ├── test_simulator.py          # Phase 1 simulator test suite (10 tests)
│   ├── test_digital_twin.py       # Phase 2 Digital Twin test suite (9 tests)
│   └── test_residual_analysis.py  # Phase 3 Residual Analysis test suite (8 tests)
└── docs/
    ├── telemetry_spec.md          # Telemetry signal & physics specification
    ├── digital_twin_model.md      # Digital Twin model equations & residual specification
    ├── residual_analysis.md       # Residual analysis, rule signatures & severity specification
    └── plots/
        ├── telemetry_comparison.png     # Simulator scenario visual plots
        ├── digital_twin_validation.png  # Digital Twin expected vs actual residual plots
        └── residual_analysis_validation.png # Anomaly score, residual signature & fault timeline plots
```

---

## 6. Running the Pipeline & Generating Data

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Python API Example
```python
from src.simulator import generate_telemetry
from src.digital_twin import DigitalTwin
from src.residual_analysis import ResidualAnalyzer

# 1. Generate telemetry (e.g., overheating scenario)
df = generate_telemetry(scenario="overheating", duration_minutes=20.0, seed=42)

# 2. Predict expected engine states and compute residuals
twin = DigitalTwin()
df_dt = twin.predict_expected_state(df)

# 3. Analyze residuals and classify faults
analyzer = ResidualAnalyzer()
df_diag = analyzer.analyze(df_dt)

# Inspect final diagnostic output
final = df_diag.iloc[-1]
print(f"Fault Type  : {final['fault_type']}")
print(f"Score       : {final['anomaly_score']} ({final['anomaly_level']})")
print(f"Severity    : {final['severity']}")
print(f"Evidence    : {final['evidence_score']}")
print(f"Contributors: {final['contributing_signals']}")
print(f"Reasoning   : {final['reasoning']}")
```

### Run Fault Detection Validation & Confusion Matrix
To execute the fault detection evaluation, confusion matrix, and generate diagnostic plots:
```bash
python scripts/validate_fault_detection.py
```

---

## 7. Running Automated Tests

Run the full pytest suite (27 test cases across Phases 1, 2, and 3):

```bash
python -m pytest tests/ -v
```
