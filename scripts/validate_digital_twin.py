"""
AeroVigil DT - Digital Twin Validation Utility
Evaluates expected-state predictions and residuals across all 5 telemetry scenarios.
Generates comprehensive visual plots comparing Actual vs Expected states & residuals.
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Ensure root workspace is in import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator import generate_telemetry
from src.digital_twin import DigitalTwin

PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "plots"))


def validate_and_plot():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Running Digital Twin Validation & Plot Generation in: {PLOTS_DIR}")

    scenarios = ["normal", "overheating", "lubrication_fault", "vibration_anomaly", "sensor_drift"]
    results = {}

    twin = DigitalTwin()

    for sc in scenarios:
        df_raw = generate_telemetry(scenario=sc, seed=42)
        res_df = twin.predict_expected_state(df_raw)
        results[sc] = res_df

        # Print summary statistics for second half of mission
        half_idx = len(res_df) // 2
        print(f"\n--- Scenario: {sc.upper()} (Second-half Residual Averages) ---")
        for sig in ["CHT", "EGT", "oil_pressure", "oil_temperature", "vibration"]:
            mean_res = res_df[f"residual_{sig}"].iloc[half_idx:].mean()
            mean_norm = res_df[f"normalized_residual_{sig}"].iloc[half_idx:].mean()
            print(f"  {sig:15s} -> Raw Residual: {mean_res:8.2f} | Norm Residual: {mean_norm:6.2f} std")

    # Generate Overview Validation Plot: Actual vs Expected across key scenarios
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    fig.suptitle("AeroVigil DT — Digital Twin Expected-State & Residual Validation", fontsize=14, fontweight="bold")

    time_sec = results["normal"]["timestamp_sec"]

    # 1. Overheating: CHT (Actual vs Expected)
    axes[0, 0].plot(time_sec, results["overheating"]["CHT"], label="Observed CHT (Overheating)", color="#d62728", linewidth=1.8)
    axes[0, 0].plot(time_sec, results["overheating"]["expected_CHT"], label="Expected CHT (Digital Twin)", color="#1f77b4", linestyle="--", linewidth=1.8)
    axes[0, 0].set_ylabel("CHT (°C)")
    axes[0, 0].set_title("1. Overheating Fault: CHT Observed vs Expected Baseline")
    axes[0, 0].legend(loc="upper left")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # 1b. Overheating: CHT Residual
    axes[0, 1].plot(time_sec, results["overheating"]["residual_CHT"], label="CHT Residual (Observed - Expected)", color="#d62728", linewidth=1.8)
    axes[0, 1].axhline(0, color="black", linestyle=":", alpha=0.7)
    axes[0, 1].set_ylabel("Residual CHT (°C)")
    axes[0, 1].set_title("1b. Overheating Fault: CHT Residual Signature (Clear +60°C Offset)")
    axes[0, 1].legend(loc="upper left")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # 2. Lubrication Fault: Oil Pressure (Actual vs Expected)
    axes[1, 0].plot(time_sec, results["lubrication_fault"]["oil_pressure"], label="Observed Oil Pressure", color="#2ca02c", linewidth=1.8)
    axes[1, 0].plot(time_sec, results["lubrication_fault"]["expected_oil_pressure"], label="Expected Oil Pressure", color="#1f77b4", linestyle="--", linewidth=1.8)
    axes[1, 0].set_ylabel("Oil Pressure (psi)")
    axes[1, 0].set_title("2. Lubrication Fault: Oil Pressure Observed vs Expected")
    axes[1, 0].legend(loc="lower left")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # 2b. Lubrication Fault: Oil Pressure Residual
    axes[1, 1].plot(time_sec, results["lubrication_fault"]["residual_oil_pressure"], label="Oil Press Residual", color="#2ca02c", linewidth=1.8)
    axes[1, 1].axhline(0, color="black", linestyle=":", alpha=0.7)
    axes[1, 1].set_ylabel("Residual Oil Press (psi)")
    axes[1, 1].set_title("2b. Lubrication Fault: Pressure Loss Residual (-34 psi Drop)")
    axes[1, 1].legend(loc="lower left")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    # 3. Vibration Anomaly: Vibration (Actual vs Expected)
    axes[2, 0].plot(time_sec, results["vibration_anomaly"]["vibration"], label="Observed Vibration", color="#9467bd", linewidth=1.8)
    axes[2, 0].plot(time_sec, results["vibration_anomaly"]["expected_vibration"], label="Expected Vibration", color="#1f77b4", linestyle="--", linewidth=1.8)
    axes[2, 0].set_ylabel("Vibration (g)")
    axes[2, 0].set_title("3. Vibration Anomaly: Observed vs Expected Baseline")
    axes[2, 0].legend(loc="upper left")
    axes[2, 0].grid(True, linestyle="--", alpha=0.5)

    # 3b. Sensor Drift: CHT vs EGT Residual Comparison (Isolated Fault Discrimination)
    axes[2, 1].plot(time_sec, results["sensor_drift"]["residual_CHT"], label="CHT Residual (Drifting Sensor)", color="#ff7f0e", linewidth=1.8)
    axes[2, 1].plot(time_sec, results["sensor_drift"]["residual_EGT"], label="EGT Residual (Normal Physics)", color="#1f77b4", linestyle=":", linewidth=1.8)
    axes[2, 1].plot(time_sec, results["sensor_drift"]["residual_oil_temperature"], label="Oil Temp Residual (Normal Physics)", color="#2ca02c", linestyle="--", linewidth=1.5)
    axes[2, 1].axhline(0, color="black", linestyle=":", alpha=0.7)
    axes[2, 1].set_ylabel("Residual Value")
    axes[2, 1].set_title("3b. Sensor Drift Isolation: CHT Residual Spikes while EGT/Oil Temp Nominal")
    axes[2, 1].legend(loc="upper left")
    axes[2, 1].grid(True, linestyle="--", alpha=0.5)

    # 4. Normal Mission: Multi-signal Residual Stability
    axes[3, 0].plot(time_sec, results["normal"]["residual_CHT"], label="Normal CHT Residual", color="#1f77b4", alpha=0.8)
    axes[3, 0].plot(time_sec, results["normal"]["residual_EGT"], label="Normal EGT Residual", color="#ff7f0e", alpha=0.8)
    axes[3, 0].plot(time_sec, results["normal"]["residual_oil_pressure"], label="Normal Oil Press Residual", color="#2ca02c", alpha=0.8)
    axes[3, 0].axhline(0, color="black", linestyle=":", alpha=0.7)
    axes[3, 0].set_ylabel("Residual Value")
    axes[3, 0].set_xlabel("Elapsed Flight Time (seconds)")
    axes[3, 0].set_title("4. Healthy Normal Mission: Small Bounded Residuals (~0 Mean)")
    axes[3, 0].legend(loc="upper right")
    axes[3, 0].grid(True, linestyle="--", alpha=0.5)

    # 4b. Normalized Residual Comparison under Overheating vs Sensor Drift
    axes[3, 1].plot(time_sec, results["overheating"]["normalized_residual_CHT"], label="Overheating: Norm CHT Residual", color="#d62728", linewidth=1.5)
    axes[3, 1].plot(time_sec, results["overheating"]["normalized_residual_EGT"], label="Overheating: Norm EGT Residual", color="#e377c2", linewidth=1.5)
    axes[3, 1].plot(time_sec, results["sensor_drift"]["normalized_residual_CHT"], label="Sensor Drift: Norm CHT Residual", color="#ff7f0e", linestyle="--", linewidth=1.5)
    axes[3, 1].plot(time_sec, results["sensor_drift"]["normalized_residual_EGT"], label="Sensor Drift: Norm EGT Residual", color="#17becf", linestyle=":", linewidth=1.5)
    axes[3, 1].axhline(0, color="black", linestyle=":", alpha=0.7)
    axes[3, 1].axhline(3.0, color="red", linestyle="--", alpha=0.5, label="3-Sigma Anomaly Threshold")
    axes[3, 1].set_ylabel("Normalized Residual (Std Devs)")
    axes[3, 1].set_xlabel("Elapsed Flight Time (seconds)")
    axes[3, 1].set_title("4b. Thermal Fault (Multi-Signal) vs Sensor Drift (Single-Signal) Discrimination")
    axes[3, 1].legend(loc="upper left")
    axes[3, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_file = os.path.join(PLOTS_DIR, "digital_twin_validation.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"\nDigital Twin Validation plot saved to: {out_file}")


if __name__ == "__main__":
    validate_and_plot()
