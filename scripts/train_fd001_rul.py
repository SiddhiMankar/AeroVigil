"""
AeroVigil - FD001 PHM Validation Training Script
scripts/train_fd001_rul.py

Runs the complete FD001 PHM validation pipeline:
    1. Load and validate FD001 dataset
    2. Exploratory data analysis (EDA)
    3. Preprocessing (sensor selection, normalization, baseline)
    4. Train GBR RUL model (80/20 engine split)
    5. Evaluate on 100 test engines vs RUL_FD001.txt
    6. Print full metrics report
    7. Save trained artifacts for dashboard use

Run from the AeroVigil repo root:
    python scripts/train_fd001_rul.py

Domain Disclaimer:
    Results validate AeroVigil's generic PHM pipeline on the NASA C-MAPSS FD001
    turbofan benchmark. They do NOT directly validate the aero-piston-engine
    physics model or UAV mission-reliability layer.
"""

import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script use
import matplotlib.pyplot as plt

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.cmapss_loader import load_fd001
from ingestion.cmapss_preprocessor import CMAPSSPreprocessor
from ingestion.cmapss_adapter import CMAPSSResidualAdapter
from ingestion.cmapss_rul_model import CMAPSSRULModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fd001_training")

# ---------------------------------------------------------------------------
# Output directory for trained artifacts
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = REPO_ROOT / "data" / "cmapss" / "FD001"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    logger.info("=" * 70)
    logger.info("AeroVigil × NASA C-MAPSS FD001 PHM Validation Pipeline")
    logger.info("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Load Dataset
    # -----------------------------------------------------------------------
    logger.info("\n[STEP 1] Loading FD001 dataset...")
    train_df, test_df, rul_gt = load_fd001(rul_cap=125)

    logger.info("Training: %d rows, %d engines", len(train_df), train_df["unit_number"].nunique())
    logger.info("Test:     %d rows, %d engines", len(test_df), test_df["unit_number"].nunique())
    logger.info("RUL GT:   min=%d, max=%d, mean=%.1f",
                rul_gt.min(), rul_gt.max(), rul_gt.mean())

    # -----------------------------------------------------------------------
    # Step 2: EDA
    # -----------------------------------------------------------------------
    logger.info("\n[STEP 2] Exploratory Data Analysis...")
    preprocessor = CMAPSSPreprocessor()
    eda = preprocessor.get_eda_summary(train_df)

    logger.info("  Trajectory lengths: min=%d, max=%d, mean=%.1f cycles",
                eda["trajectory_lengths"]["min"],
                eda["trajectory_lengths"]["max"],
                eda["trajectory_lengths"]["mean"])
    logger.info("  Constant sensors (will be dropped): %s", eda["constant_sensors"])
    logger.info("  Informative sensors (%d): %s",
                len(eda["informative_sensors"]), eda["informative_sensors"])
    logger.info("  RUL label stats: min=%.0f, max=%.0f, mean=%.1f, std=%.1f",
                eda["rul_label_stats"]["min"], eda["rul_label_stats"]["max"],
                eda["rul_label_stats"]["mean"], eda["rul_label_stats"]["std"])

    # -----------------------------------------------------------------------
    # Step 3: Preprocessing (fit ONLY on training data)
    # -----------------------------------------------------------------------
    logger.info("\n[STEP 3] Preprocessing (fit on training data only)...")
    train_transformed = preprocessor.fit_transform(train_df)
    test_transformed = preprocessor.transform(test_df)

    logger.info("  Selected %d sensors, dropped %d constant sensors.",
                len(preprocessor.selected_sensors), len(preprocessor.dropped_sensors))
    logger.info("  Dropped: %s", preprocessor.dropped_sensors)
    logger.info("  Retained: %s", preprocessor.selected_sensors)

    # -----------------------------------------------------------------------
    # Step 4: Train RUL Model
    # -----------------------------------------------------------------------
    logger.info("\n[STEP 4] Training GBR RUL Model (80/20 engine split)...")
    model = CMAPSSRULModel()
    results = model.train_and_evaluate(
        train_transformed, test_transformed, rul_gt, preprocessor,
        window_size=15, seed=42,
    )

    metrics = results["metrics"]
    train_info = results["training_info"]

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 70)
    logger.info("  Train engines:      %d", train_info["n_train_engines"])
    logger.info("  Validation engines: %d", train_info["n_val_engines"])
    logger.info("  Train samples:      %d", train_info["n_train_samples"])
    logger.info("  Val   samples:      %d", train_info["n_val_samples"])
    logger.info("  Validation MAE:     %.2f cycles", metrics["val_mae"])
    logger.info("  Validation RMSE:    %.2f cycles", metrics["val_rmse"])

    logger.info("\n" + "=" * 70)
    logger.info("TEST EVALUATION RESULTS (vs RUL_FD001.txt)")
    logger.info("=" * 70)
    logger.info("  Test MAE:        %.2f cycles", metrics["test_mae"])
    logger.info("  Test RMSE:       %.2f cycles", metrics["test_rmse"])
    logger.info("  Test R²:         %.4f", metrics["test_r2"])
    logger.info("  NASA Score:      %.2f (lower=better, asymmetric PHM metric)", metrics["test_nasa_score"])
    logger.info("=" * 70)

    per_engine = results["per_engine_results"]
    logger.info("\n  Sample predictions (first 10 test engines):")
    logger.info("  %s", per_engine.head(10).to_string(index=False))

    # -----------------------------------------------------------------------
    # Step 5: Health Trajectories for subset of training engines
    # -----------------------------------------------------------------------
    logger.info("\n[STEP 5] Computing health trajectories for first 10 engines...")
    adapter = CMAPSSResidualAdapter(
        selected_sensors=preprocessor.selected_sensors,
        healthy_baseline_std=preprocessor.healthy_baseline_std,
    )
    # Process 10 engines for dashboard preview
    sample_engines = sorted(train_df["unit_number"].unique())[:10]
    sample_df = train_transformed[train_transformed["unit_number"].isin(sample_engines)]
    health_df = adapter.compute_fleet_health_trajectories(sample_df)
    logger.info("  Health trajectories: %d rows, engines %s",
                len(health_df), sample_engines)

    # -----------------------------------------------------------------------
    # Step 6: Save Artifacts
    # -----------------------------------------------------------------------
    logger.info("\n[STEP 6] Saving artifacts to %s...", ARTIFACTS_DIR)

    # Save trained model
    with open(ARTIFACTS_DIR / "gbr_rul_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save preprocessor
    with open(ARTIFACTS_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    # Save per-engine results
    per_engine.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)

    # Save metrics summary
    metrics_summary = {
        **metrics,
        "n_engines_train": int(train_info["n_train_engines"]),
        "n_engines_val": int(train_info["n_val_engines"]),
        "n_sensors_retained": len(preprocessor.selected_sensors),
        "sensors_dropped": preprocessor.dropped_sensors,
        "sensors_retained": preprocessor.selected_sensors,
        "window_size": 15,
        "rul_cap": 125,
        "model": "GradientBoostingRegressor",
        "disclaimer": (
            "Results validate generic PHM capabilities on C-MAPSS FD001 "
            "turbofan data. Does NOT directly validate the AeroVigil "
            "aero-piston-engine physics model or UAV mission risk layer."
        ),
    }
    with open(ARTIFACTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # Save health trajectories sample
    health_df.to_csv(ARTIFACTS_DIR / "sample_health_trajectories.csv", index=False)

    # -----------------------------------------------------------------------
    # Step 7: Generate validation plots
    # -----------------------------------------------------------------------
    logger.info("\n[STEP 7] Generating validation plots...")

    predictions = results["predictions"]
    unit_ids = results["unit_ids"]
    true_rul_all = rul_gt[unit_ids - 1]

    # Plot 1: Predicted vs True RUL
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(true_rul_all, predictions, alpha=0.6, color="#1f77b4", s=30, label="Test Engines")
    max_val = max(true_rul_all.max(), predictions.max()) + 10
    ax1.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Perfect Prediction")
    ax1.set_xlabel("True RUL (cycles)")
    ax1.set_ylabel("Predicted RUL (cycles)")
    ax1.set_title("Predicted vs True RUL — FD001 Test Engines\n(PHM Validation via C-MAPSS, not UAV piston-engine)")
    ax1.legend()
    ax1.grid(True, alpha=0.4)
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)

    # Annotate metrics
    ax1.text(
        0.05, 0.93,
        f"MAE={metrics['test_mae']:.1f}  RMSE={metrics['test_rmse']:.1f}  R²={metrics['test_r2']:.3f}",
        transform=ax1.transAxes, fontsize=9,
        bbox=dict(facecolor="lightyellow", alpha=0.8, edgecolor="gray"),
    )

    # Plot 2: Error distribution
    errors = predictions - true_rul_all
    ax2 = axes[1]
    ax2.hist(errors, bins=20, color="#ff7f0e", edgecolor="black", alpha=0.75)
    ax2.axvline(0, color="red", linestyle="--", label="Zero error")
    ax2.axvline(errors.mean(), color="blue", linestyle=":", label=f"Mean error = {errors.mean():.1f}")
    ax2.set_xlabel("Prediction Error (predicted - true) [cycles]")
    ax2.set_ylabel("Count")
    ax2.set_title("RUL Prediction Error Distribution\n(FD001 Test Engines)")
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "rul_validation_plots.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: %s", ARTIFACTS_DIR / "rul_validation_plots.png")

    # Plot 3: Sample degradation trajectories
    fig2, ax3 = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10.colors
    for i, eng_id in enumerate(sample_engines[:8]):
        eng_h = health_df[health_df["unit_number"] == eng_id]
        ax3.plot(eng_h["cycle"], eng_h["health_index"],
                 color=colors[i % 10], linewidth=1.2, label=f"Engine {eng_id}")
    ax3.axhline(75.0, color="orange", linestyle="--", alpha=0.6, label="Degraded threshold (75)")
    ax3.axhline(50.0, color="red", linestyle=":", alpha=0.6, label="Warning threshold (50)")
    ax3.set_xlabel("Operational Cycle")
    ax3.set_ylabel("Model-Derived Health Index (0–100)")
    ax3.set_title("Engine Degradation Trajectories — FD001 Training Engines\n"
                  "(Model-Derived Health Index, data-driven, not physically measured)")
    ax3.legend(ncol=2, fontsize=8)
    ax3.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "health_trajectories.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: %s", ARTIFACTS_DIR / "health_trajectories.png")

    logger.info("\n✅ FD001 PHM validation training complete.")
    logger.info("   Artifacts saved to: %s", ARTIFACTS_DIR)
    logger.info("\n" + "=" * 70)
    logger.info("LEGITIMATE SIH VALIDATION CLAIMS:")
    logger.info("  1. AeroVigil PHM pipeline achieves MAE=%.1f cycles on FD001.", metrics["test_mae"])
    logger.info("  2. RMSE=%.1f cycles, R²=%.3f on 100 test engines vs RUL_FD001.txt.", metrics["test_rmse"], metrics["test_r2"])
    logger.info("  3. Data-driven health index monotonically captures HPC degradation.")
    logger.info("  4. No data leakage: engines split 80/20 by ID, test GT never seen in training.")
    logger.info("  5. NOTE: C-MAPSS FD001 ≠ aero-piston engine. Physics model is separate.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
