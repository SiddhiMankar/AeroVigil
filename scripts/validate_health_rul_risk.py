"""
AeroVigil DT - Integrated Health, RUL & Mission Risk Validation Utility
Phase 4 Component

Executes the full end-to-end AeroVigil DT pipeline across all 5 telemetry scenarios.
Generates evaluation tables and comprehensive diagnostic plots saved to docs/plots/.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repo root is on path if running as standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator import generate_telemetry
from src.health_risk_pipeline import run_pipeline
import config

PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "plots"))


def validate_and_plot():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Running Integrated Health, RUL & Mission Risk Validation in: {PLOTS_DIR}\n")

    scenarios = ["normal", "overheating", "lubrication_fault", "vibration_anomaly", "sensor_drift"]
    results = {}
    summary_table = []

    for sc in scenarios:
        df_raw = generate_telemetry(scenario=sc, seed=42)
        df_processed = run_pipeline(df_raw)
        results[sc] = df_processed

        final_row = df_processed.iloc[-1]
        min_health = df_processed["health_index"].min()

        # Find minimum RUL reached during mission
        valid_ruls = df_processed["rul_minutes"].dropna()
        min_rul = valid_ruls.min() if not valid_ruls.empty else None

        summary_table.append(
            {
                "Scenario": sc.upper(),
                "Min Health": min_health,
                "Final Health": final_row["health_index"],
                "Health State": final_row["health_state"],
                "Min RUL (min)": round(min_rul, 1) if min_rul is not None else "STABLE",
                "Final Risk Score": final_row["mission_risk_score"],
                "Final Risk Level": final_row["mission_risk_level"],
                "Final Advisory": final_row["mission_recommendation"],
            }
        )

    print("=" * 110)
    print("AEROVIGIL DT — PHASE 4 INTEGRATED PIPELINE EVALUATION SUMMARY")
    print("=" * 110)
    summary_df = pd.DataFrame(summary_table)
    print(summary_df.to_string(index=False))
    print("=" * 110 + "\n")

    time_sec = results["normal"]["timestamp_sec"]

    colors = {
        "normal": "#1f77b4",
        "overheating": "#d62728",
        "lubrication_fault": "#2ca02c",
        "vibration_anomaly": "#9467bd",
        "sensor_drift": "#ff7f0e",
    }

    # Plot 1: Health Index Comparison
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for sc in scenarios:
        ax1.plot(time_sec, results[sc]["health_index"], label=sc.upper(), color=colors[sc], linewidth=2.0)

    ax1.axhline(90.0, color="green", linestyle="--", alpha=0.5, label="Healthy Threshold (90)")
    ax1.axhline(75.0, color="orange", linestyle="--", alpha=0.5, label="Degraded Threshold (75)")
    ax1.axhline(50.0, color="darkorange", linestyle="--", alpha=0.5, label="Warning Threshold (50)")
    ax1.axhline(25.0, color="red", linestyle=":", alpha=0.7, label="Critical Threshold (25)")
    ax1.set_ylabel("Engine Health Index (0-100)")
    ax1.set_xlabel("Elapsed Flight Time (seconds)")
    ax1.set_title("AeroVigil DT — Engine Health Index Trajectories Across Scenarios", fontweight="bold")
    ax1.legend(loc="lower left")
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot1_path = os.path.join(PLOTS_DIR, "health_index_comparison.png")
    plt.savefig(plot1_path, dpi=150)
    plt.close()

    # Plot 2: RUL Trajectory
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for sc in ["overheating", "lubrication_fault", "vibration_anomaly", "sensor_drift"]:
        df_sc = results[sc]
        ax2.plot(df_sc["timestamp_sec"], df_sc["rul_minutes"], label=sc.upper(), color=colors[sc], linewidth=2.0)

    ax2.axhline(10.0, color="orange", linestyle="--", alpha=0.6, label="Low RUL Threshold (10 min)")
    ax2.axhline(3.0, color="red", linestyle=":", alpha=0.7, label="Critical RUL Threshold (3 min)")
    ax2.set_ylabel("Estimated Prototype RUL (minutes)")
    ax2.set_xlabel("Elapsed Flight Time (seconds)")
    ax2.set_title("AeroVigil DT — Prototype RUL Trajectories Under Degradation", fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot2_path = os.path.join(PLOTS_DIR, "rul_trajectory.png")
    plt.savefig(plot2_path, dpi=150)
    plt.close()

    # Plot 3: Mission Risk Timeline
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    for sc in scenarios:
        ax3.plot(time_sec, results[sc]["mission_risk_score"], label=sc.upper(), color=colors[sc], linewidth=2.0)

    ax3.axhline(20.0, color="green", linestyle="--", alpha=0.5, label="Low Risk (<20)")
    ax3.axhline(45.0, color="orange", linestyle="--", alpha=0.5, label="Moderate Risk (<45)")
    ax3.axhline(70.0, color="red", linestyle=":", alpha=0.7, label="High Risk (<70)")
    ax3.set_ylabel("Mission Risk Score (0-100)")
    ax3.set_xlabel("Elapsed Flight Time (seconds)")
    ax3.set_title("AeroVigil DT — Mission Risk Score Timelines Across Scenarios", fontweight="bold")
    ax3.legend(loc="upper left")
    ax3.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot3_path = os.path.join(PLOTS_DIR, "mission_risk_timeline.png")
    plt.savefig(plot3_path, dpi=150)
    plt.close()

    # Plot 4: Integrated 4-Panel Engine State Overview
    fig4, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig4.suptitle("AeroVigil DT — Integrated End-to-End Engine State Overview", fontsize=14, fontweight="bold")

    # 4a. Health Index
    for sc in scenarios:
        axes[0].plot(time_sec, results[sc]["health_index"], label=sc.upper(), color=colors[sc], linewidth=1.8)
    axes[0].axhline(25.0, color="red", linestyle=":", label="Critical Threshold")
    axes[0].set_ylabel("Health Index")
    axes[0].set_title("1. Engine Health Index Progression")
    axes[0].legend(loc="lower left")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 4b. Fault Severity
    sev_map = {"NORMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    for sc in scenarios:
        sev_nums = [sev_map[s] for s in results[sc]["severity"]]
        axes[1].plot(time_sec, sev_nums, label=sc.upper(), color=colors[sc], linewidth=1.8)
    axes[1].set_yticks([0, 1, 2, 3, 4])
    axes[1].set_yticklabels(["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
    axes[1].set_ylabel("Fault Severity")
    axes[1].set_title("2. Diagnostic Fault Severity Classification")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 4c. RUL Estimates
    for sc in ["overheating", "lubrication_fault", "vibration_anomaly", "sensor_drift"]:
        axes[2].plot(time_sec, results[sc]["rul_minutes"], label=sc.upper(), color=colors[sc], linewidth=1.8)
    axes[2].set_ylabel("RUL (minutes)")
    axes[2].set_title("3. Prototype RUL Estimates During Degradation")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # 4d. Mission Risk & Recommendation State
    for sc in scenarios:
        axes[3].plot(time_sec, results[sc]["mission_risk_score"], label=sc.upper(), color=colors[sc], linewidth=1.8)
    axes[3].set_ylabel("Mission Risk Score")
    axes[3].set_xlabel("Elapsed Flight Time (seconds)")
    axes[3].set_title("4. Operational Mission Risk Assessment")
    axes[3].legend(loc="upper left")
    axes[3].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plot4_path = os.path.join(PLOTS_DIR, "integrated_engine_state.png")
    plt.savefig(plot4_path, dpi=150)
    plt.close()

    print(f"Validation plots saved successfully in: {PLOTS_DIR}")


if __name__ == "__main__":
    validate_and_plot()
