# AeroVigil DT — Evaluator Demo Walkthrough Guide

This guide provides a 2-minute step-by-step walkthrough for evaluating the **AeroVigil DT** system.

---

## 1. Fast Command-Line Demo

To run the complete end-to-end pipeline in your terminal without launching a web browser:

```bash
python scripts/run_demo.py
```

This processes the 20-minute **Overheating Thermal Fault** scenario (Seed 42) and outputs a concise pipeline summary.

---

## 2. Interactive Streamlit Dashboard Walkthrough

### Step 1: Launch Dashboard
```bash
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

### Step 2: Observe Default Scenario (Overheating Thermal Fault)
The dashboard defaults to:
* **Scenario**: Overheating Thermal Fault
* **Duration**: 20 minutes (1200 seconds)
* **Random Seed**: 42

Observe the 8 key pipeline stages in action:

1. **KPI Cards (Top Bar)**: Observe Engine Health Index ($10.9 / 100$ `CRITICAL`), Est. RUL, Mission Risk ($100.0 / 100$ `CRITICAL`), and Classified Fault (`OVERHEATING`).
2. **Operational Advisory Banner**: Displays `PROTOTYPE ADVISORY: CONSIDER MISSION ABORT`.
3. **Live Telemetry Panel (Section 1)**: Inspect real-time sensor metrics (`RPM`, `CHT`, `EGT`, `Oil Pressure`, `Oil Temp`, `Vibration`, etc.) with physical units.
4. **Digital Twin Panel (Section 2)**: Compare **Observed CHT/EGT** (Red/Purple) vs **Digital Twin Expected State** (Blue dashed lines). Notice how observed CHT diverges significantly above expected physics state starting at $t=360\text{s}$.
5. **Residual & Anomaly Detection (Section 3)**: Inspect normalized residuals ($z_{\text{CHT}}, z_{\text{EGT}}$) crossing the $3.0\sigma$ threshold line, and read the explainable diagnostic reasoning.
6. **Health Index & RUL Trajectories (Section 4)**: Watch the smooth Health Index deterioration curve drop below the Critical threshold ($25$).
7. **Mission Risk Timeline (Section 5)**: Track the Flight-Phase Aware Mission Risk score rising to $100.0$.
8. **Interactive Mission Replay Slider**: Drag the slider in the sidebar back to $t = 200\text{s}$ to verify the system correctly reflects healthy nominal state before fault onset!

---

### Step 3: Test Sensor Drift Isolation (Sensor vs Physical Engine Fault)
1. In the sidebar dropdown, select **CHT Sensor Drift Bias**.
2. Notice the key technical distinction:
   * Classified Fault changes to **`SENSOR_DRIFT`**.
   * Mission Risk drops to **`MODERATE` (32.4)**.
   * Prototype Advisory changes to **`INSPECT_AT_NEXT_OPPORTUNITY`** (instead of aborting the mission!).
   * Section 3 shows that **only CHT residual** is elevated ($+14.2\sigma$), while correlated physical signals ($EGT$, $OilTemp$) remain completely nominal ($< 0.3\sigma$).

---

### Step 4: Inspect Cross-Scenario Evaluation Matrix
Click the **Cross-Scenario Evaluation Matrix** tab in Section 5 to view the benchmark comparison table across all 5 flight scenarios (`NORMAL`, `OVERHEATING`, `LUBRICATION_FAULT`, `VIBRATION_ANOMALY`, `SENSOR_DRIFT`).

---

## 3. Running Automated Tests

To execute the complete 47-test pytest suite:

```bash
python -m pytest tests/ -v
```
