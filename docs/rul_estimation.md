# AeroVigil DT — Prototype RUL Estimation Specification

## 1. Overview & Purpose

The **Prototype RUL Estimator** calculates the Remaining Useful Life (RUL) of the simulated engine in seconds and minutes.

> **IMPORTANT DISCLAIMER**: Real engine run-to-failure datasets are unavailable in this project. RUL is demonstrated using controlled synthetic degradation trajectories. This estimator is a **prototype indicator** and is **NOT** claimed to represent certified aircraft engine RUL algorithms or physical component wear curves.

---

## 2. Linear Slope Degradation Projection

The estimator tracks the recent degradation slope of the Health Index ($\frac{dH}{dt}$) over a sliding window ($T_{\text{window}} = 45\text{ s}$):

$$\frac{dH}{dt} = \text{LinearSlope}\left(\mathbf{t}_{\text{window}}, \mathbf{H}_{\text{window}}\right)$$

### RUL Calculation Logic
1. **Stable / Healthy State**:
   If $H(t) \ge 92.0$ or $\frac{dH}{dt} \ge -0.01\text{ pts/sec}$:
   * $\text{degradation\_rate} \approx 0.0$
   * $\text{RUL}_{\text{sec}} = \text{NaN}$
   * $\text{RUL}_{\text{min}} = \text{NaN}$
   * $\text{rul\_status} = \text{`STABLE`}$ (or `UNAVAILABLE`)

2. **Active Degradation State**:
   If $\frac{dH}{dt} < -0.01\text{ pts/sec}$:
   * Remaining Health Buffer: $\Delta H = \max\left(0.0, H(t) - H_{\text{crit}}\right)$ where $H_{\text{crit}} = 25.0$.
   * Raw RUL Estimate:
     $$\text{RUL}_{\text{sec}} = \frac{H(t) - H_{\text{crit}}}{\left|\frac{dH}{dt}\right|}$$
   * Exponential low-pass filter applied to prevent numerical jitter.

---

## 3. RUL Status Categories

* **`STABLE`**: Health Index is high ($\ge 92$) and no sustained degradation rate detected.
* **`DEGRADING`**: Active health decline detected ($\text{RUL}_{\text{min}} \ge 10.0\text{ min}$).
* **`LOW_RUL`**: Warning remaining time ($3.0\text{ min} \le \text{RUL}_{\text{min}} < 10.0\text{ min}$).
* **`CRITICAL_RUL`**: Critical remaining time ($\text{RUL}_{\text{min}} < 3.0\text{ min}$ or $H(t) \le 25.0$).
* **`UNAVAILABLE`**: Insufficient history or zero slope.

---

## 4. Engineering Safeguards

* **No Negative RUL**: RUL estimates are strictly bounded $\ge 0.0$.
* **No Fabricated Drops**: Healthy operation explicitly returns `STABLE` / `NaN` without inventing false countdown timers.

---

## 5. API Example

```python
from src.rul_estimator import RULEstimator

rul_est = RULEstimator()
df_augmented = rul_est.estimate(df_with_health_index)

# Output fields:
# - df_augmented["degradation_rate"] (float pts/sec)
# - df_augmented["rul_seconds"]      (float or NaN)
# - df_augmented["rul_minutes"]      (float or NaN)
# - df_augmented["rul_status"]       (str: STABLE, DEGRADING, LOW_RUL, CRITICAL_RUL, UNAVAILABLE)
```
