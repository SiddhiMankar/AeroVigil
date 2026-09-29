# AeroVigil DT — Engine Health Index Specification

## 1. Overview & Purpose

The **Engine Health Index** ($H \in [0.0, 100.0]$) provides a transparent, physics-informed, continuous indicator of an aero-piston engine's operational integrity.

> **DISCLAIMER**: The Engine Health Index is a **synthetic prototype indicator** designed for technical feasibility demonstration (SIH 2026). It does **not** represent certified aerospace engine health limits.

---

## 2. Mathematical Formulation & Penalties

The Health Index is derived by subtracting a composite penalty from 100.0, followed by exponential moving average smoothing:

$$\text{Health Penalty} = w_{\text{anom}} \cdot P_{\text{anom}} + w_{\text{sev}} \cdot P_{\text{sev}} + w_{\text{flt}} \cdot P_{\text{flt}} + w_{\text{pers}} \cdot P_{\text{pers}}$$

$$H_{\text{instant}}(t) = \max\left(0.0, \min\left(100.0, 100.0 - \text{Health Penalty}\right)\right)$$

$$H_{\text{smooth}}(t) = H_{\text{smooth}}(t-1) + \alpha \cdot \left(H_{\text{instant}}(t) - H_{\text{smooth}}(t-1)\right)$$

Where:
* $w_{\text{anom}} = 0.40$ (Anomaly Score component)
* $w_{\text{sev}} = 0.30$ (Diagnostic Severity component)
* $w_{\text{flt}} = 0.15$ (Fault Type Evidence component)
* $w_{\text{pers}} = 0.15$ (Multi-signal Persistence component)
* $\alpha = 0.15$ (Smoothing coefficient)

---

## 3. Health State Categories

| Health Index Range | Health State Category | Operational Meaning |
|---|---|---|
| **90.0 – 100.0** | `HEALTHY` | Nominal engine operation with normal thermodynamic noise |
| **75.0 – <90.0** | `DEGRADED` | Minor residual elevation; non-critical degradation |
| **50.0 – <75.0** | `WARNING` | Moderate anomaly or isolated sensor failure |
| **25.0 – <50.0** | `SEVERE` | Significant physical thermal or lubrication failure |
| **0.0 – <25.0** | `CRITICAL` | Severe multi-signal engine breakdown |

---

## 4. Health Trend Indicator

Computes slope $\frac{dH}{dt}$ over a 15-sample sliding window:
* $\frac{dH}{dt} < -0.03\text{ pts/sec} \longrightarrow$ **`DEGRADING`**
* $\frac{dH}{dt} > +0.03\text{ pts/sec} \longrightarrow$ **`IMPROVING`**
* Otherwise $\longrightarrow$ **`STABLE`**

---

## 5. API Example

```python
from src.health_index import EngineHealthIndex

health_calc = EngineHealthIndex()
df_augmented = health_calc.calculate(df_with_residuals)

# Output fields:
# - df_augmented["health_index"]   (float 0.0 - 100.0)
# - df_augmented["health_state"]   (str: HEALTHY, DEGRADED, WARNING, SEVERE, CRITICAL)
# - df_augmented["health_penalty"] (float)
# - df_augmented["health_trend"]   (str: STABLE, DEGRADING, IMPROVING)
```
