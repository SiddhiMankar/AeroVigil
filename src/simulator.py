"""
AeroVigil DT - Synthetic Aero-Piston Engine Telemetry Simulator
Phase 1 Foundation Component

Generates physics-correlated synthetic time-series telemetry for a MALE UAV aero-piston engine
under normal mission profiles and controlled fault injection scenarios.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import sys
import os

# Ensure repo root is on path if running as standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


class EngineSimulator:
    """
    Synthetic Aero-Piston Engine Telemetry Generator.
    Models flight mission phases and correlated thermodynamic & mechanical sensors.
    """

    def __init__(self, seed: Optional[int] = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _generate_mission_profile(
        self, duration_minutes: float, sample_rate_hz: float
    ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
        """
        Generates timeline, mission phase labels, smooth throttle trajectory, and ambient temp.
        """
        total_seconds = duration_minutes * 60.0
        n_steps = int(total_seconds * sample_rate_hz)
        time_sec = np.linspace(0, total_seconds, n_steps, endpoint=False)

        # Build phase transition boundaries
        phase_names = []
        target_throttles = np.zeros(n_steps)
        alt_temp_offsets = np.zeros(n_steps)

        cum_fraction = 0.0
        phase_nodes = []

        for p_info in config.MISSION_PHASES:
            start_frac = cum_fraction
            cum_fraction += p_info["duration_fraction"]
            end_frac = min(1.0, cum_fraction)

            start_idx = int(start_frac * n_steps)
            end_idx = int(end_frac * n_steps) if end_frac < 1.0 else n_steps

            phase_nodes.append((start_idx, end_idx, p_info["name"], p_info["throttle_target"], p_info["alt_temp_offset"]))

        # Assign phase labels and smooth step interpolation for throttle & temp
        labels = []
        raw_throttle_targets = np.zeros(n_steps)
        raw_temp_offsets = np.zeros(n_steps)

        for start_idx, end_idx, name, thr_target, temp_offset in phase_nodes:
            for idx in range(start_idx, min(end_idx, n_steps)):
                labels.append(name)
                raw_throttle_targets[idx] = thr_target
                raw_temp_offsets[idx] = temp_offset

        # Fill any missing tail labels
        while len(labels) < n_steps:
            labels.append(config.MISSION_PHASES[-1]["name"])

        # Smooth throttle transitions using moving average / low-pass filter
        kernel_size = max(5, int(15 * sample_rate_hz))
        kernel = np.ones(kernel_size) / kernel_size
        smooth_throttle = np.convolve(raw_throttle_targets, kernel, mode="same")
        smooth_temp_offset = np.convolve(raw_temp_offsets, kernel, mode="same")

        # Add minor maneuvering dynamics during MANEUVER phase
        for i in range(n_steps):
            if labels[i] == "MANEUVER":
                maneuver_variation = 0.08 * np.sin(2 * np.pi * i / (60 * sample_rate_hz))
                smooth_throttle[i] = np.clip(smooth_throttle[i] + maneuver_variation, 0.2, 0.95)

        # Ensure throttle bounds [0, 1]
        smooth_throttle = np.clip(smooth_throttle, 0.05, 1.0)

        # Base ambient temp at sea level (18 °C) + altitude offset + noise
        base_ambient = 18.0
        ambient_temp = base_ambient + smooth_temp_offset + self.rng.normal(0, 0.1, n_steps)

        return time_sec, labels, smooth_throttle, ambient_temp

    def simulate(
        self,
        scenario: str = "normal",
        duration_minutes: float = 20.0,
        sample_rate_hz: float = 1.0,
        drift_sensor: str = "CHT",
        fault_start_fraction: float = 0.3,
        start_time_iso: str = "2026-03-30T10:00:00Z",
    ) -> pd.DataFrame:
        """
        Simulate engine telemetry time-series dataset.

        Parameters:
        -----------
        scenario : str
            Scenario name: 'normal', 'overheating', 'lubrication_fault', 'vibration_anomaly', 'sensor_drift'
        duration_minutes : float
            Total flight duration in minutes (default: 20.0)
        sample_rate_hz : float
            Sampling frequency in Hz (default: 1.0)
        drift_sensor : str
            Target sensor for 'sensor_drift' scenario (default: 'CHT')
        fault_start_fraction : float
            Fraction of mission where fault begins developing (default: 0.3)
        start_time_iso : str
            ISO format start timestamp string

        Returns:
        --------
        pd.DataFrame containing simulated telemetry signals
        """
        if scenario not in config.FAULT_SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Valid options: {config.FAULT_SCENARIOS}")

        time_sec, mission_phases, throttle, ambient_temp = self._generate_mission_profile(
            duration_minutes, sample_rate_hz
        )
        n_steps = len(time_sec)

        # Allocate sensor arrays
        rpm = np.zeros(n_steps)
        fuel_flow = np.zeros(n_steps)
        egt = np.zeros(n_steps)
        cht = np.zeros(n_steps)
        oil_temp = np.zeros(n_steps)
        oil_press = np.zeros(n_steps)
        vibration = np.zeros(n_steps)
        battery = np.zeros(n_steps)

        # Initial thermodynamic states
        curr_rpm = 0.0
        curr_egt = ambient_temp[0] + 20.0
        curr_cht = ambient_temp[0] + 25.0
        curr_oil_temp = ambient_temp[0] + 15.0
        curr_oil_press = 0.0

        # Dynamics parameters (first-order low pass coefficients per step at 1 Hz)
        dt = 1.0 / sample_rate_hz
        alpha_rpm = 1.0 - np.exp(-dt / 1.5)      # 1.5s time constant
        alpha_egt = 1.0 - np.exp(-dt / 3.0)      # 3.0s time constant
        alpha_cht = 1.0 - np.exp(-dt / 35.0)     # 35s time constant
        alpha_oil_t = 1.0 - np.exp(-dt / 70.0)   # 70s time constant
        alpha_oil_p = 1.0 - np.exp(-dt / 2.0)    # 2.0s time constant

        # Pre-generate noise sequences for reproducibility
        noise_rpm = self.rng.normal(0, 8.0, n_steps)
        noise_ff = self.rng.normal(0, 0.15, n_steps)
        noise_egt = self.rng.normal(0, 1.2, n_steps)
        noise_cht = self.rng.normal(0, 0.4, n_steps)
        noise_oil_t = self.rng.normal(0, 0.15, n_steps)
        noise_oil_p = self.rng.normal(0, 0.25, n_steps)
        noise_vib = self.rng.normal(0, 0.012, n_steps)
        noise_batt = self.rng.normal(0, 0.03, n_steps)

        # Pre-compute fault progress trajectory p(t) in [0, 1]
        fault_start_idx = int(fault_start_fraction * n_steps)
        fault_progress = np.zeros(n_steps)
        for i in range(fault_start_idx, n_steps):
            fault_progress[i] = (i - fault_start_idx) / max(1, (n_steps - fault_start_idx))

        # Simulation loop
        for i in range(n_steps):
            phase = mission_phases[i]
            th = throttle[i]
            amb = ambient_temp[i]

            # 1. RPM Dynamics
            if phase == "STARTUP" and i < int(15 * sample_rate_hz):
                target_rpm = 600.0 + i * (600.0 / (15 * sample_rate_hz))
            else:
                target_rpm = 1200.0 + 4400.0 * (th ** 1.05)

            # Fault impact on RPM instability for vibration_anomaly
            rpm_fault_jitter = 0.0
            if scenario == "vibration_anomaly":
                p = fault_progress[i]
                rpm_fault_jitter = self.rng.normal(0, 60.0 * (p ** 1.5))

            curr_rpm = curr_rpm + alpha_rpm * (target_rpm - curr_rpm) + noise_rpm[i] + rpm_fault_jitter
            curr_rpm = max(0.0, curr_rpm)
            rpm[i] = curr_rpm

            # 2. Fuel Flow (L/h)
            rpm_ratio = curr_rpm / 5600.0
            target_ff = 3.2 + 38.0 * (rpm_ratio ** 1.25) * (0.35 + 0.65 * th)
            fuel_flow[i] = max(0.0, target_ff + noise_ff[i])

            # 3. EGT Dynamics (°C)
            target_egt = 380.0 + 440.0 * th + 0.6 * (curr_rpm / 100.0) + 1.2 * amb
            if scenario == "overheating":
                p = fault_progress[i]
                target_egt += 85.0 * (p ** 1.1)

            curr_egt = curr_egt + alpha_egt * (target_egt - curr_egt) + noise_egt[i]
            egt[i] = curr_egt

            # 4. CHT Dynamics (°C)
            # Cooling airflow effect reduces CHT at higher airspeed/RPM
            cooling_factor = 1.0 - 0.12 * rpm_ratio
            target_cht = 105.0 + 95.0 * th * cooling_factor + 0.8 * amb
            if scenario == "overheating":
                p = fault_progress[i]
                target_cht += 65.0 * (p ** 1.2)
            elif scenario == "lubrication_fault":
                p = fault_progress[i]
                target_cht += 12.0 * (p ** 1.3)  # secondary friction heat

            curr_cht = curr_cht + alpha_cht * (target_cht - curr_cht) + noise_cht[i]
            cht[i] = curr_cht

            # 5. Oil Temperature (°C)
            target_oil_t = 60.0 + 38.0 * th + 0.22 * (curr_cht - 90.0) + 0.4 * amb
            if scenario == "overheating":
                p = fault_progress[i]
                target_oil_t += 28.0 * (p ** 1.2)
            elif scenario == "lubrication_fault":
                p = fault_progress[i]
                target_oil_t += 36.0 * (p ** 1.2)  # major oil temp rise due to friction

            curr_oil_temp = curr_oil_temp + alpha_oil_t * (target_oil_t - curr_oil_temp) + noise_oil_t[i]
            oil_temp[i] = curr_oil_temp

            # 6. Oil Pressure (psi)
            # Mechanical pump drives pressure with RPM; oil heating thins viscosity and drops pressure
            viscosity_drop = 0.18 * max(0.0, curr_oil_temp - 75.0)
            target_oil_p = 22.0 + 38.0 * rpm_ratio - viscosity_drop
            if scenario == "lubrication_fault":
                p = fault_progress[i]
                target_oil_p -= 34.0 * (p ** 1.1)  # severe pressure loss

            curr_oil_press = curr_oil_press + alpha_oil_p * (target_oil_p - curr_oil_press) + noise_oil_p[i]
            oil_press[i] = max(5.0, curr_oil_press)

            # 7. Vibration (g)
            target_vib = 0.08 + 0.20 * (rpm_ratio ** 2) + 0.04 * th
            if scenario == "vibration_anomaly":
                p = fault_progress[i]
                target_vib += 0.70 * (p ** 1.3)

            vibration[i] = max(0.01, target_vib + noise_vib[i])

            # 8. Battery Voltage (V)
            if curr_rpm < 800.0:
                # Cranking / discharging
                target_batt = 12.2 - 0.5 * (1.0 if phase == "STARTUP" else 0.0)
            else:
                # Alternator charging
                target_batt = 13.95 + 0.05 * th
            battery[i] = target_batt + noise_batt[i]

        # Apply Sensor Drift (Bias strictly on ONE selected sensor without changing true underlying physics)
        if scenario == "sensor_drift":
            drift_bias = 65.0 * fault_progress
            if drift_sensor == "CHT":
                cht += drift_bias
            elif drift_sensor == "EGT":
                egt += drift_bias
            elif drift_sensor == "oil_pressure":
                oil_press -= drift_bias * 0.4
            elif drift_sensor == "oil_temperature":
                oil_temp += drift_bias * 0.5
            elif drift_sensor == "RPM":
                rpm += drift_bias * 10.0
            elif drift_sensor == "vibration":
                vibration += drift_bias * 0.01
            else:
                cht += drift_bias

        # Generate timestamps
        base_dt = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
        timestamps = [(base_dt + timedelta(seconds=float(ts))).isoformat() for ts in time_sec]

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "timestamp_sec": np.round(time_sec, 2),
                "mission_phase": mission_phases,
                "throttle": np.round(throttle, 4),
                "ambient_temperature": np.round(ambient_temp, 2),
                "RPM": np.round(rpm, 1),
                "CHT": np.round(cht, 2),
                "EGT": np.round(egt, 2),
                "oil_pressure": np.round(oil_press, 2),
                "oil_temperature": np.round(oil_temp, 2),
                "fuel_flow": np.round(fuel_flow, 2),
                "vibration": np.round(vibration, 4),
                "battery_voltage": np.round(battery, 2),
            }
        )

        return df


def generate_telemetry(
    scenario: str = "normal",
    duration_minutes: float = 20.0,
    sample_rate_hz: float = 1.0,
    seed: Optional[int] = 42,
    drift_sensor: str = "CHT",
    fault_start_fraction: float = 0.3,
) -> pd.DataFrame:
    """
    Convenience functional API for generating synthetic telemetry datasets.
    """
    sim = EngineSimulator(seed=seed)
    return sim.simulate(
        scenario=scenario,
        duration_minutes=duration_minutes,
        sample_rate_hz=sample_rate_hz,
        drift_sensor=drift_sensor,
        fault_start_fraction=fault_start_fraction,
    )
