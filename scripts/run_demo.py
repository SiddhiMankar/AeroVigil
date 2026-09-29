"""
AeroVigil DT - Reproducible Demonstration Script
Phase 6 Component

Runs the complete end-to-end AeroVigil DT pipeline on the Overheating scenario (Seed 42, 20 min)
and outputs a concise, evaluator-friendly terminal summary.
"""

import os
import sys
import pandas as pd

# Ensure repo root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator import generate_telemetry
from src.health_risk_pipeline import run_pipeline
import config


def main():
    print("=" * 90)
    print("AEROVIGIL DT -- REPRODUCIBLE END-TO-END DEMONSTRATION")
    print("AI-Enabled Real-Time Digital Twin System for UAV Aero-Piston Engines (SIH 2026)")
    print("=" * 90)

    scenario = "overheating"
    duration_mins = 20.0
    seed = 42

    print(f"\nRunning Production Pipeline on Scenario: {scenario.upper()}")
    print(f"  Duration   : {duration_mins:.0f} minutes (1200 seconds at 1 Hz)")
    print(f"  Random Seed: {seed} (Deterministic Reproducibility)")
    print("  Processing : Telemetry -> Digital Twin -> Residual Analysis -> Health -> RUL -> Risk...")

    # Step 1: Generate Telemetry
    df_raw = generate_telemetry(scenario=scenario, duration_minutes=duration_mins, seed=seed)

    # Step 2: Execute Integrated Pipeline
    df_out = run_pipeline(df_raw)

    final = df_out.iloc[-1]
    peak_anomaly = df_out["anomaly_score"].max()
    min_health = df_out["health_index"].min()

    rul_m = final["rul_minutes"]
    rul_str = f"{rul_m:.1f} minutes" if not pd.isna(rul_m) else "N/A (At Critical Threshold)"

    reasoning_clean = str(final["reasoning"]).replace("\u03c3", " std")

    print("\n" + "-" * 90)
    print("PIPELINE EVALUATION SUMMARY OUTPUT")
    print("-" * 90)
    print(f"  Total Samples Processed  : {len(df_out)} time-steps")
    print(f"  Final Classified Fault   : {final['fault_type']}")
    print(f"  Diagnostic Severity      : {final['severity']}")
    print(f"  Diagnostic Evidence      : {final['evidence_score']*100:.0f}% ({final['evidence_score']:.2f})")
    print(f"  Peak Anomaly Score       : {peak_anomaly:.1f} / 100")
    print(f"  Minimum Health Index     : {min_health:.1f} / 100")
    print(f"  Final Health Index       : {final['health_index']:.1f} ({final['health_state']})")
    print(f"  Prototype RUL Estimate   : {rul_str} ({final['rul_status']})")
    print(f"  Final Mission Risk Score : {final['mission_risk_score']:.1f} / 100 ({final['mission_risk_level']})")
    print(f"  Active Mission Phase     : {final['mission_phase']}")
    print(f"  PROTOTYPE ADVISORY       : {final['mission_recommendation']}")
    print(f"  Contributing Signals     : {final['contributing_signals']}")
    print(f"  Diagnostic Reasoning     : {reasoning_clean}")
    print("-" * 90)

    print("\n[SUCCESS] AeroVigil DT end-to-end pipeline demonstration executed successfully!")
    print("To launch the interactive dashboard, run: streamlit run app.py\n")


if __name__ == "__main__":
    main()
