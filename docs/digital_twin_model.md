# AeroVigil DT — Physics-Based Digital Twin Model Specification

## 1. Concept & Purpose

In **AeroVigil DT**, the **Digital Twin** is a software surrogate model that estimates the nominal, expected state of an aero-piston engine in real time based strictly on healthy physical operating conditions:

$$\text{Operating Conditions } (\text{throttle}, T_{amb}, \text{phase}) \longrightarrow \text{Physics Surrogate Model} \longrightarrow \text{Expected Engine State}$$

By comparing the observed telemetry against the Digital Twin's expected state, the system computes **residuals**:

$$\text{Residual}_i(t) = \text{Observed}_i(t) - \text{Expected}_i(t)$$

$$\text{Normalized Residual}_i(t) = \frac{\text{Observed}_i(t) - \text{Expected}_i(t)}{\sigma_{i, \text{nominal}}}$$

These residuals serve as the primary input feature vector for **Phase 3 (Residual Analysis & Anomaly Detection)**.

---

## 2. Model Inputs & Outputs

### Inputs (Operating Conditions Only)
* `throttle` — Throttle lever position ($0.0 \le \text{throttle} \le 1.0$)
* `ambient_temperature` — Outside air temperature ($T_{amb}$, °C)
* `mission_phase` — Flight profile phase (`STARTUP`, `TAKEOFF`, `CLIMB`, `CRUISE`, `MANEUVER`, `DESCENT`, `LANDING`)
* `timestamp_sec` / `dt` — Elapsed time / time step

> **CRITICAL RULE**: The Digital Twin model does **NOT** receive or inspect scenario/fault labels (`overheating`, `lubrication_fault`, etc.). It operates strictly as a healthy baseline estimator.

### Outputs
* **Expected State Columns**: `expected_RPM`, `expected_CHT`, `expected_EGT`, `expected_oil_pressure`, `expected_oil_temperature`, `expected_fuel_flow`, `expected_vibration`, `expected_battery_voltage`
* **Raw Residual Columns**: `residual_RPM`, `residual_CHT`, `residual_EGT`, `residual_oil_pressure`, `residual_oil_temperature`, `residual_fuel_flow`, `residual_vibration`, `residual_battery_voltage`
* **Normalized Residual Columns**: `normalized_residual_<signal>` (scaled by nominal standard deviations $\sigma_{\text{nominal}}$)

---

## 3. Reduced-Order Physics Surrogate Relationships

1. **Expected RPM**:
   $$RPM_{\text{target}} = 1200 + 4400 \cdot \text{throttle}^{1.05}$$
   Smooth first-order inertia spool-up ($\tau = 1.5\text{ s}$).

2. **Expected Fuel Flow**:
   $$FF_{\text{exp}} = 3.2 + 38.0 \cdot \left(\frac{RPM_{\text{exp}}}{5600}\right)^{1.25} \cdot (0.35 + 0.65 \cdot \text{throttle})$$

3. **Expected EGT**:
   $$EGT_{\text{target}} = 380 + 440 \cdot \text{throttle} + 0.6 \cdot \left(\frac{RPM_{\text{exp}}}{100}\right) + 1.2 \cdot T_{amb}$$
   Dynamic thermal filter ($\tau = 3.0\text{ s}$).

4. **Expected CHT**:
   $$\text{Airflow Cooling Factor} = 1.0 - 0.12 \cdot \left(\frac{RPM_{\text{exp}}}{5600}\right)$$
   $$CHT_{\text{target}} = 105 + 95 \cdot \text{throttle} \cdot \text{Cooling Factor} + 0.8 \cdot T_{amb}$$
   Thermal mass inertia filter ($\tau = 35\text{ s}$).

5. **Expected Oil Temperature**:
   $$OilTemp_{\text{target}} = 60 + 38 \cdot \text{throttle} + 0.22 \cdot (CHT_{\text{exp}} - 90) + 0.4 \cdot T_{amb}$$
   High thermal inertia filter ($\tau = 70\text{ s}$).

6. **Expected Oil Pressure**:
   $$\text{Viscosity Loss} = 0.18 \cdot \max(0, OilTemp_{\text{exp}} - 75.0)$$
   $$OilPress_{\text{exp}} = 22.0 + 38.0 \cdot \left(\frac{RPM_{\text{exp}}}{5600}\right) - \text{Viscosity Loss}$$

7. **Expected Vibration**:
   $$Vib_{\text{exp}} = 0.08 + 0.20 \cdot \left(\frac{RPM_{\text{exp}}}{5600}\right)^2 + 0.04 \cdot \text{throttle}$$

8. **Expected Battery Voltage**:
   $$\text{If } RPM_{\text{exp}} < 800\text{ RPM: } 12.2\text{ V}; \quad \text{If } RPM_{\text{exp}} \ge 800\text{ RPM: } 13.95 + 0.05 \cdot \text{throttle} \text{ V}$$

---

## 4. State Dynamics & Time-Series Smoothing

To reflect true physical thermal inertia, state update variables ($\mathbf{x}_t$) evolve dynamically across consecutive time steps using exponential low-pass filtering:

$$\mathbf{x}_t = \mathbf{x}_{t-1} + \left(1 - e^{-\Delta t / \tau}\right) \left(\mathbf{x}_{\text{target}} - \mathbf{x}_{t-1}\right)$$

This prevents instant non-physical step jumps in expected thermal variables ($CHT$, $EGT$, $OilTemp$) when throttle commands change abruptly.

---

## 5. Physical Fault vs Sensor Drift Discrimination

The Digital Twin's independent expected-state estimator provides clear mathematical signatures to distinguish physical engine faults from electrical sensor drift:

| Scenario | $\text{Residual}_{CHT}$ | $\text{Residual}_{EGT}$ | $\text{Residual}_{OilTemp}$ | $\text{Residual}_{OilPress}$ | Structural Interpretation |
|---|---|---|---|---|---|
| **`normal`** | $\approx 0$ | $\approx 0$ | $\approx 0$ | $\approx 0$ | Healthy nominal operation |
| **`overheating`** | $\gg 0$ | $\gg 0$ | $> 0$ | $\approx 0$ | **Multi-Signal Physical Fault** (Thermal elevation) |
| **`lubrication_fault`** | $> 0$ | $\approx 0$ | $\gg 0$ | $\ll 0$ | **Multi-Signal Physical Fault** (Hydraulic loss & friction) |
| **`vibration_anomaly`** | $\approx 0$ | $\approx 0$ | $\approx 0$ | $\approx 0$ | **Mechanical Structural Fault** ($\text{Residual}_{Vib} \gg 0$) |
| **`sensor_drift` (CHT)** | $\gg 0$ | $\approx 0$ | $\approx 0$ | $\approx 0$ | **Isolated Single-Sensor Fault** |

In `sensor_drift`, because only the targeted sensor (CHT) deviates while correlated signals ($EGT$, $OilTemp$) remain near zero residual, downstream anomaly detection can conclusively classify the issue as a **sensor failure**, avoiding false engine shutdown commands!

---

## 6. Interface for Phase 3 (Residual Analysis)

Phase 3 can consume the Digital Twin via batch or online API:

```python
from src.digital_twin import DigitalTwin

# Batch processing
twin = DigitalTwin()
augmented_df = twin.predict_expected_state(telemetry_df)

# Online streaming point processing
step_dict = twin.predict_step(
    throttle=0.65,
    ambient_temperature=18.0,
    mission_phase="CRUISE",
    observed_telemetry={"CHT": 195.0, "EGT": 720.0, "oil_pressure": 52.0}
)
```

---

## 7. Model Limitations

1. **Surrogate Model**: Reduced-order physics representation designed for prototype demonstration.
2. **Fixed Ambient Dynamics**: Does not model extreme ambient weather anomalies (icing, dust storms).
3. **Calibrated Parameters**: Constants are tuned for the prototype MALE UAV aero-piston engine baseline.
