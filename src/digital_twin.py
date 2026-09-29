"""
AeroVigil DT - Physics-Based Digital Twin (Expected-State Model)
Phase 2 Foundation Component

Provides a reduced-order physics-informed surrogate estimator for an aero-piston engine.
Predicts nominal expected engine states based on operating conditions (throttle, ambient temp,
flight phase) and computes residuals between observed telemetry and expected states.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import sys
import os

# Ensure repo root is on path if running as standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


class DigitalTwin:
    """
    Independent Physics-Informed Digital Twin Expected-State Estimator.

    Computes nominal thermodynamic and mechanical states of a healthy aero-piston engine
    from operational inputs (throttle, ambient temperature, flight phase).
    Calculates residuals: Residual = Observed - Expected.
    """

    def __init__(self, sample_rate_hz: float = 1.0):
        self.sample_rate_hz = sample_rate_hz
        self.reset()

    def reset(self):
        """Resets internal state variables of the Digital Twin filters."""
        self.curr_rpm: float = 0.0
        self.curr_egt: float = 25.0
        self.curr_cht: float = 30.0
        self.curr_oil_temp: float = 25.0
        self.curr_oil_press: float = 0.0
        self.step_counter: int = 0
        self.last_timestamp: Optional[str] = None
        self.last_phase: str = "STARTUP"

    def predict_step(
        self,
        throttle: float,
        ambient_temperature: float,
        mission_phase: str = "CRUISE",
        observed_telemetry: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Single-step online estimation of expected engine state.

        Parameters:
        -----------
        throttle : float
            Engine throttle position (0.0 to 1.0)
        ambient_temperature : float
            Ambient outside air temperature (°C)
        mission_phase : str
            Current flight mission phase
        observed_telemetry : Optional[Dict[str, float]]
            Optional observed telemetry readings to compute instantaneous residuals.

        Returns:
        --------
        Dict containing expected states (and residuals if observed_telemetry is provided).
        """
        dt = 1.0 / self.sample_rate_hz

        # Filter time constants for physics surrogate dynamics
        alpha_rpm = 1.0 - np.exp(-dt / 1.5)
        alpha_egt = 1.0 - np.exp(-dt / 3.0)
        alpha_cht = 1.0 - np.exp(-dt / 35.0)
        alpha_oil_t = 1.0 - np.exp(-dt / 70.0)
        alpha_oil_p = 1.0 - np.exp(-dt / 2.0)

        # 1. Expected RPM
        if mission_phase == "STARTUP" and self.step_counter < int(15 * self.sample_rate_hz):
            target_rpm = 600.0 + self.step_counter * (600.0 / (15 * self.sample_rate_hz))
        else:
            target_rpm = 1200.0 + 4400.0 * (max(0.0, min(1.0, throttle)) ** 1.05)

        self.curr_rpm = self.curr_rpm + alpha_rpm * (target_rpm - self.curr_rpm)
        exp_rpm = max(0.0, self.curr_rpm)

        # 2. Expected Fuel Flow (L/h)
        rpm_ratio = exp_rpm / 5600.0
        exp_fuel_flow = 3.2 + 38.0 * (rpm_ratio ** 1.25) * (0.35 + 0.65 * throttle)

        # 3. Expected EGT (°C)
        target_egt = 380.0 + 440.0 * throttle + 0.6 * (exp_rpm / 100.0) + 1.2 * ambient_temperature
        self.curr_egt = self.curr_egt + alpha_egt * (target_egt - self.curr_egt)
        exp_egt = self.curr_egt

        # 4. Expected CHT (°C)
        cooling_factor = 1.0 - 0.12 * rpm_ratio
        target_cht = 105.0 + 95.0 * throttle * cooling_factor + 0.8 * ambient_temperature
        self.curr_cht = self.curr_cht + alpha_cht * (target_cht - self.curr_cht)
        exp_cht = self.curr_cht

        # 5. Expected Oil Temperature (°C)
        target_oil_t = 60.0 + 38.0 * throttle + 0.22 * (exp_cht - 90.0) + 0.4 * ambient_temperature
        self.curr_oil_temp = self.curr_oil_temp + alpha_oil_t * (target_oil_t - self.curr_oil_temp)
        exp_oil_temp = self.curr_oil_temp

        # 6. Expected Oil Pressure (psi)
        viscosity_drop = 0.18 * max(0.0, exp_oil_temp - 75.0)
        target_oil_p = 22.0 + 38.0 * rpm_ratio - viscosity_drop
        self.curr_oil_press = self.curr_oil_press + alpha_oil_p * (target_oil_p - self.curr_oil_press)
        exp_oil_press = max(5.0, self.curr_oil_press)

        # 7. Expected Vibration (g)
        exp_vibration = 0.08 + 0.20 * (rpm_ratio ** 2) + 0.04 * throttle

        # 8. Expected Battery Voltage (V)
        if exp_rpm < 800.0:
            exp_battery = 12.2 - 0.5 * (1.0 if mission_phase == "STARTUP" else 0.0)
        else:
            exp_battery = 13.95 + 0.05 * throttle

        self.step_counter += 1

        result = {
            "expected_RPM": round(float(exp_rpm), 1),
            "expected_CHT": round(float(exp_cht), 2),
            "expected_EGT": round(float(exp_egt), 2),
            "expected_oil_pressure": round(float(exp_oil_press), 2),
            "expected_oil_temperature": round(float(exp_oil_temp), 2),
            "expected_fuel_flow": round(float(exp_fuel_flow), 2),
            "expected_vibration": round(float(exp_vibration), 4),
            "expected_battery_voltage": round(float(exp_battery), 2),
        }

        # Calculate residuals if observed telemetry is provided
        if observed_telemetry is not None:
            for sig in config.MODELED_SIGNALS:
                if sig in observed_telemetry:
                    obs_val = observed_telemetry[sig]
                    exp_val = result[f"expected_{sig}"]
                    res = obs_val - exp_val
                    norm_res = res / config.NOMINAL_STD_DEV.get(sig, 1.0)
                    result[f"residual_{sig}"] = round(float(res), 4)
                    result[f"normalized_residual_{sig}"] = round(float(norm_res), 4)

        return result

    def predict_expected_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch estimation of expected engine state and residuals for a telemetry DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            Telemetry dataframe containing at minimum: 'throttle', 'ambient_temperature',
            'mission_phase', and observed signal columns.

        Returns:
        --------
        pd.DataFrame
            Copy of input DataFrame augmented with expected_<signal>, residual_<signal>,
            and normalized_residual_<signal> columns.
        """
        # Ensure reset before batch prediction
        self.reset()

        result_df = df.copy()

        n_rows = len(df)
        expected_dict = {sig: np.zeros(n_rows) for sig in config.MODELED_SIGNALS}

        throttles = df["throttle"].values
        ambient_temps = df["ambient_temperature"].values
        phases = df["mission_phase"].values if "mission_phase" in df.columns else ["CRUISE"] * n_rows

        for i in range(n_rows):
            th = throttles[i]
            amb = ambient_temps[i]
            ph = phases[i]

            step_res = self.predict_step(throttle=th, ambient_temperature=amb, mission_phase=ph)

            for sig in config.MODELED_SIGNALS:
                expected_dict[sig][i] = step_res[f"expected_{sig}"]

        # Attach expected columns to result DataFrame
        for sig in config.MODELED_SIGNALS:
            exp_col = f"expected_{sig}"
            result_df[exp_col] = expected_dict[sig]

            # Compute residuals if observed signal is present in input DataFrame
            if sig in df.columns:
                res_col = f"residual_{sig}"
                norm_res_col = f"normalized_residual_{sig}"
                obs_vals = df[sig].values
                exp_vals = expected_dict[sig]

                residuals = obs_vals - exp_vals
                norm_dev = config.NOMINAL_STD_DEV.get(sig, 1.0)
                norm_residuals = residuals / norm_dev

                result_df[res_col] = np.round(residuals, 4)
                result_df[norm_res_col] = np.round(norm_residuals, 4)

        return result_df

    def get_state(self) -> Dict[str, Union[float, int, str]]:
        """
        Returns snapshot of current Digital Twin state variables.
        """
        return {
            "step_counter": self.step_counter,
            "curr_rpm": round(self.curr_rpm, 1),
            "curr_cht": round(self.curr_cht, 2),
            "curr_egt": round(self.curr_egt, 2),
            "curr_oil_temp": round(self.curr_oil_temp, 2),
            "curr_oil_press": round(self.curr_oil_press, 2),
        }
