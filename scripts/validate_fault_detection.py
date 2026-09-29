"""
AeroVigil DT - Residual Analysis & Fault Detection Validation Utility
Phase 3 Component

Evaluates ResidualAnalyzer across all 5 telemetry scenarios.
Generates evaluation metrics, detection onset times, a confusion matrix,
and visual validation plots for anomaly scores, residual signatures, and fault timelines.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure root workspace is in import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator import generate_telemetry
from src.digital_twin import DigitalTwin
from src.residual_analysis import ResidualAnalyzer
import config

PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "plots"))


def validate_and_plot():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Running Residual Analysis & Fault Detection Validation in: {PLOTS_DIR}\n")

    scenarios = ["normal", "overheating", "lubrication_fault", "vibration_anomaly", "sensor_drift"]
    scenario_label_map = {
        "normal": "NORMAL",
        "overheating": "OVERHEATING",
        "lubrication_fault": "LUBRICATION_FAULT",
        "vibration_anomaly": "VIBRATION_ANOMALY",
        "sensor_drift": "SENSOR_DRIFT",
    }

    results = {}
    twin = DigitalTwin()
    analyzer = ResidualAnalyzer()

    confusion_matrix = pd.DataFrame(0, index=config.FAULT_TYPES, columns=config.FAULT_TYPES)

    summary_table = []

    for sc in scenarios:
        df_raw = generate_telemetry(scenario=sc, seed=42)
        df_dt = twin.predict_expected_state(df_raw)
        df_diag = analyzer.analyze(df_dt)
        results[sc] = df_diag

        expected_class = scenario_label_map[sc]

        # Calculate detection onset (first second where fault_type != NORMAL)
        fault_mask = df_diag["fault_type"] != "NORMAL"
        if fault_mask.any():
            onset_idx = fault_mask.idxmax()
            onset_sec = df_diag["timestamp_sec"].iloc[onset_idx]
        else:
            onset_sec = None

        final_row = df_diag.iloc[-1]
        pred_class = final_row["fault_type"]

        confusion_matrix.loc[expected_class, pred_class] += 1

        summary_table.append(
            {
                "Scenario": sc,
                "Expected Fault": expected_class,
                "Detected Fault": pred_class,
                "Onset Time (s)": onset_sec if onset_sec is not None else "N/A (Healthy)",
                "Final Score": final_row["anomaly_score"],
                "Final Severity": final_row["severity"],
                "Evidence Score": final_row["evidence_score"],
                "Contributing Signals": final_row["contributing_signals"],
            }
        )

    print("=" * 100)
    print("AEROVIGIL DT — PHASE 3 FAULT DETECTION EVALUATION SUMMARY")
    print("=" * 100)
    summary_df = pd.DataFrame(summary_table)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX (Ground Truth Scenario vs Detected Fault Class)")
    print("=" * 60)
    print(confusion_matrix)

    # Calculate overall classification accuracy across the 5 scenario runs
    correct = sum(confusion_matrix.loc[c, c] for c in scenario_label_map.values())
    total = len(scenarios)
    accuracy = (correct / total) * 100.0
    print(f"\nScenario Fault Classification Accuracy: {correct}/{total} ({accuracy:.1f}%)\n")

    # Generate Visualization Plots
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("AeroVigil DT — Residual Analysis & Anomaly Detection Validation", fontsize=14, fontweight="bold")

    time_sec = results["normal"]["timestamp_sec"]

    # 1. Global Anomaly Score Timeline Across All Scenarios
    for sc in scenarios:
        color = {
            "normal": "#1f77b4",
            "overheating": "#d62728",
            "lubrication_fault": "#2ca02c",
            "vibration_anomaly": "#9467bd",
            "sensor_drift": "#ff7f0e",
        }[sc]
        axes[0].plot(time_sec, results[sc]["anomaly_score"], label=f"{sc.upper()}", color=color, linewidth=1.8)

    axes[0].axhline(15.0, color="orange", linestyle="--", alpha=0.6, label="Warning Threshold (15)")
    axes[0].axhline(35.0, color="red", linestyle="--", alpha=0.6, label="Anomalous Threshold (35)")
    axes[0].axhline(65.0, color="darkred", linestyle=":", alpha=0.7, label="Critical Threshold (65)")
    axes[0].set_ylabel("Anomaly Score (0-100)")
    axes[0].set_title("1. Global Anomaly Score Progression Across Mission Profile")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Overheating Residual Signatures
    df_over = results["overheating"]
    axes[1].plot(time_sec, df_over["normalized_residual_CHT"], label="CHT Norm Residual", color="#d62728", linewidth=1.8)
    axes[1].plot(time_sec, df_over["normalized_residual_EGT"], label="EGT Norm Residual", color="#e377c2", linewidth=1.8)
    axes[1].plot(time_sec, df_over["normalized_residual_oil_temperature"], label="Oil Temp Norm Residual", color="#ff7f0e", linewidth=1.5)
    axes[1].axhline(3.0, color="red", linestyle=":", label="3σ Anomaly Threshold")
    axes[1].set_ylabel("Normalized Residual (σ)")
    axes[1].set_title("2. Overheating Fault: Multi-Signal Thermal Residual Signature (CHT & EGT > 3σ)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Sensor Drift Discrimination (Single CHT vs Nominal EGT/Oil Temp)
    df_drift = results["sensor_drift"]
    axes[2].plot(time_sec, df_drift["normalized_residual_CHT"], label="CHT Norm Residual (Drifting Sensor)", color="#ff7f0e", linewidth=1.8)
    axes[2].plot(time_sec, df_drift["normalized_residual_EGT"], label="EGT Norm Residual (Nominal Engine)", color="#1f77b4", linestyle=":", linewidth=1.8)
    axes[2].plot(time_sec, df_drift["normalized_residual_oil_temperature"], label="Oil Temp Norm Residual (Nominal)", color="#2ca02c", linestyle="--", linewidth=1.5)
    axes[2].axhline(3.0, color="red", linestyle=":", label="3σ Anomaly Threshold")
    axes[2].set_ylabel("Normalized Residual (σ)")
    axes[2].set_title("3. Sensor Drift Discrimination: Isolated CHT Bias while Correlated Signals Nominal")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # 4. Detected Severity Timeline (Overheating vs Lubrication vs Normal)
    sev_map = {"NORMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    for sc in ["normal", "overheating", "lubrication_fault", "sensor_drift"]:
        sev_nums = [sev_map[s] for s in results[sc]["severity"]]
        color = {"normal": "#1f77b4", "overheating": "#d62728", "lubrication_fault": "#2ca02c", "sensor_drift": "#ff7f0e"}[sc]
        axes[3].plot(time_sec, sev_nums, label=sc.upper(), color=color, linewidth=1.8)

    axes[3].set_yticks([0, 1, 2, 3, 4])
    axes[3].set_yticklabels(["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
    axes[3].set_ylabel("Diagnostic Severity")
    axes[3].set_xlabel("Elapsed Flight Time (seconds)")
    axes[3].set_title("4. Diagnostic Fault Severity Transition Timeline")
    axes[3].legend(loc="upper left")
    axes[3].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_file = os.path.join(PLOTS_DIR, "residual_analysis_validation.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Validation plot saved to: {out_file}")


if __name__ == "__main__":
    validate_and_plot()
