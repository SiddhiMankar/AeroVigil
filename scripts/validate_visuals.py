"""
AeroVigil DT - Visual Validation Utility
Generates comparison plots of telemetry signals for development validation.
Compares Normal vs Overheating, Normal vs Lubrication Fault, and Normal vs Sensor Drift.
"""

import os
import sys
import matplotlib.pyplot as plt

# Ensure root workspace is in import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator import generate_telemetry

PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "plots"))

def generate_visual_validation():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Generating visual validation plots in: {PLOTS_DIR}")

    # Generate telemetry datasets
    df_normal = generate_telemetry(scenario="normal", seed=42)
    df_overheating = generate_telemetry(scenario="overheating", seed=42)
    df_lubrication = generate_telemetry(scenario="lubrication_fault", seed=42)
    df_vibration = generate_telemetry(scenario="vibration_anomaly", seed=42)
    df_drift = generate_telemetry(scenario="sensor_drift", seed=42)

    time_sec = df_normal["timestamp_sec"]

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    fig.suptitle("AeroVigil DT — Synthetic Telemetry & Fault Injection Visual Validation", fontsize=14, fontweight="bold")

    # 1. RPM Comparison
    axes[0].plot(time_sec, df_normal["RPM"], label="Normal Baseline", color="#1f77b4", linewidth=1.5)
    axes[0].plot(time_sec, df_vibration["RPM"], label="Vibration Anomaly (RPM jitter)", color="#9467bd", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("RPM (rev/min)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. CHT Comparison (Normal vs Overheating vs Sensor Drift)
    axes[1].plot(time_sec, df_normal["CHT"], label="Normal Baseline", color="#1f77b4", linewidth=1.5)
    axes[1].plot(time_sec, df_overheating["CHT"], label="Overheating Fault", color="#d62728", linewidth=1.5)
    axes[1].plot(time_sec, df_drift["CHT"], label="CHT Sensor Drift (Isolated Bias)", color="#ff7f0e", linestyle="--", linewidth=1.5)
    axes[1].set_ylabel("CHT (°C)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. EGT Comparison (Normal vs Overheating vs Sensor Drift)
    axes[2].plot(time_sec, df_normal["EGT"], label="Normal Baseline", color="#1f77b4", linewidth=1.5)
    axes[2].plot(time_sec, df_overheating["EGT"], label="Overheating Fault (Rises with CHT)", color="#d62728", linewidth=1.5)
    axes[2].plot(time_sec, df_drift["EGT"], label="CHT Sensor Drift (EGT Unchanged)", color="#ff7f0e", linestyle=":", linewidth=1.5)
    axes[2].set_ylabel("EGT (°C)")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # 4. Oil Pressure Comparison (Normal vs Lubrication Fault)
    axes[3].plot(time_sec, df_normal["oil_pressure"], label="Normal Baseline", color="#1f77b4", linewidth=1.5)
    axes[3].plot(time_sec, df_lubrication["oil_pressure"], label="Lubrication Fault (Pressure Loss)", color="#2ca02c", linewidth=1.5)
    axes[3].set_ylabel("Oil Pressure (psi)")
    axes[3].legend(loc="lower left")
    axes[3].grid(True, linestyle="--", alpha=0.5)

    # 5. Vibration Comparison (Normal vs Vibration Anomaly)
    axes[4].plot(time_sec, df_normal["vibration"], label="Normal Baseline", color="#1f77b4", linewidth=1.5)
    axes[4].plot(time_sec, df_vibration["vibration"], label="Vibration Anomaly", color="#9467bd", linewidth=1.5)
    axes[4].set_ylabel("Vibration (g)")
    axes[4].set_xlabel("Elapsed Time (seconds)")
    axes[4].legend(loc="upper left")
    axes[4].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_file = os.path.join(PLOTS_DIR, "telemetry_comparison.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Validation plot saved to: {out_file}")

if __name__ == "__main__":
    generate_visual_validation()
