# AeroVigil DT — Mission Risk & Advisory Model Specification

## 1. Overview & Purpose

The **Mission Risk Model** combines engine health index, remaining useful life (RUL) estimates, diagnostic severity, fault type classification, and flight mission phase vulnerability into an operational risk score ($0\text{--}100$) and prototype decision advisory.

> **IMPORTANT DISCLAIMER**: The Mission Risk Model provides **prototype simulation advisories** for demonstration purposes (SIH 2026). It does **NOT** represent certified flight-control commands and cannot autonomously command a physical UAV.

---

## 2. Risk Calculation & Flight Phase Sensitivity

$$\text{Weighted Base Risk} = 0.35 \cdot R_{\text{health}} + 0.40 \cdot R_{\text{severity}} + 0.25 \cdot R_{\text{RUL}}$$

$$\text{Mission Risk Score} = \min\left(100.0, \text{Weighted Base Risk} \cdot \text{Phase\_Multiplier} \cdot \text{Fault\_Factor}\right)$$

### Flight Phase Sensitivity Multipliers

| Mission Phase | Sensitivity Multiplier | Rationale |
|---|---|---|
| **`STARTUP`** | 0.8x | Ground phase; engine can be aborted on ground with minimal risk |
| **`TAKEOFF`** | 1.5x | Critical flight regime; maximum engine load & thrust dependency |
| **`CLIMB`** | 1.3x | High power requirement; elevated risk during altitude gain |
| **`CRUISE`** | 1.0x | Nominal baseline flight profile |
| **`MANEUVER`** | 1.2x | Tactical loading; elevated thermal & mechanical stress |
| **`DESCENT`** | 1.2x | Gliding altitude available, but engine restart required |
| **`LANDING`** | 1.4x | Critical touchdown phase; engine response mandatory |

### Fault Type Factors

* **`NORMAL`**: 0.0x
* **`SENSOR_DRIFT`**: 0.55x (Reduced risk factor because underlying physical engine thermal/hydraulic state is healthy!)
* **`VIBRATION_ANOMALY`**: 1.05x
* **`OVERHEATING`**: 1.30x (Severe thermal threat)
* **`LUBRICATION_FAULT`**: 1.40x (Critical hydraulic threat; risk of immediate engine seizure)

---

## 3. Risk Levels & Operational Advisories

| Mission Risk Score | Risk Level | Prototype Operational Advisory |
|---|---|---|
| **0.0 – <20.0** | `LOW` | **`CONTINUE_MONITORING`** |
| **20.0 – <45.0** | `MODERATE` | **`INCREASE_MONITORING`** |
| **45.0 – <70.0** | `HIGH` | **`INSPECT_AT_NEXT_OPPORTUNITY`** |
| **70.0 – 100.0** | `CRITICAL` | **`CONSIDER_MISSION_ABORT`** |

> **Sensor Drift Specific Advisory**: If `fault_type == "SENSOR_DRIFT"` and risk level is non-critical, the system outputs **`INSPECT_AT_NEXT_OPPORTUNITY`** instead of forcing an unnecessary mission abort!

---

## 4. API Example

```python
from src.mission_risk import MissionRiskEstimator

risk_est = MissionRiskEstimator()
df_augmented = risk_est.assess(df_with_rul)

# Output fields:
# - df_augmented["mission_risk_score"]          (float 0.0 - 100.0)
# - df_augmented["mission_risk_level"]          (str: LOW, MODERATE, HIGH, CRITICAL)
# - df_augmented["mission_recommendation"]      (str: CONTINUE_MONITORING, INCREASE_MONITORING, INSPECT_AT_NEXT_OPPORTUNITY, CONSIDER_MISSION_ABORT)
```
