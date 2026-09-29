"""
AeroVigil DT - Data Generation Script
Generates representative CSV telemetry datasets for all 5 simulated scenarios.
"""

import os
import sys

# Ensure root workspace is in import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator import generate_telemetry
import config

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Generating telemetry datasets in: {DATA_DIR}")

    scenarios = [
        ("normal", "normal.csv"),
        ("overheating", "overheating.csv"),
        ("lubrication_fault", "lubrication_fault.csv"),
        ("vibration_anomaly", "vibration_fault.csv"),
        ("sensor_drift", "sensor_drift.csv"),
    ]

    for scenario_name, filename in scenarios:
        filepath = os.path.join(DATA_DIR, filename)
        print(f"Generating scenario '{scenario_name}' -> {filename} ...")
        df = generate_telemetry(
            scenario=scenario_name,
            duration_minutes=20.0,
            sample_rate_hz=1.0,
            seed=42
        )
        df.to_csv(filepath, index=False)
        print(f"  Saved {len(df)} rows to {filepath}")

    print("Data generation complete!")

if __name__ == "__main__":
    main()
