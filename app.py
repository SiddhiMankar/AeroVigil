"""
AeroVigil DT — AI-Enabled Real-Time Aero-Engine Digital Twin System
Phase 5 Streamlit Application

Real-Time Monitoring, Digital Twin Visualization, Residual Anomaly Detection,
Explainable Fault Diagnosis, Health Index Tracking, Prototype RUL Estimation,
and Mission Risk Advisories for UAV Aero-Piston Engines.

Tab 1: AeroVigil DT (Synthetic simulation, piston-engine digital twin)
Tab 2: FD001 PHM Validation (NASA C-MAPSS benchmark, generic PHM validation)
       NOTE: FD001 validates PHM capabilities only. It does NOT validate
       the aero-piston-engine physics model or UAV mission-reliability layer.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure root workspace is on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.dashboard_utils import (
    SCENARIO_DISPLAY_NAMES,
    SCENARIO_NAME_MAP,
    extract_snapshot_at_index,
    generate_scenario_comparison,
    run_cached_simulation,
)
import config

# FD001 PHM Validation imports (isolated from core AeroVigil pipeline)
try:
    import json
    import pickle
    from pathlib import Path
    _FD001_ARTIFACTS_DIR = Path(__file__).resolve().parent / "data" / "cmapss" / "FD001"
    _FD001_AVAILABLE = _FD001_ARTIFACTS_DIR.exists() and ((_FD001_ARTIFACTS_DIR / "gbr_rul_model.pkl").exists())
except Exception:
    _FD001_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroVigil DT — Aero-Engine Digital Twin",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 14px;
        color: #6B7280;
        margin-bottom: 15px;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-title {
        font-size: 12px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
        margin: 4px 0;
    }
    .kpi-badge {
        display: inline-block;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .badge-normal { background-color: #DCFCE7; color: #166534; }
    .badge-warning { background-color: #FEF9C3; color: #854D0E; }
    .badge-anomalous { background-color: #FFEDD5; color: #9A3412; }
    .badge-critical { background-color: #FEE2E2; color: #991B1B; }
    .advisory-box {
        padding: 12px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 14px;
        margin-top: 10px;
        text-align: center;
    }
    .adv-continue { background-color: #DCFCE7; border-left: 5px solid #16A34A; color: #15803D; }
    .adv-increase { background-color: #FEF9C3; border-left: 5px solid #CA8A04; color: #A16207; }
    .adv-inspect { background-color: #FFEDD5; border-left: 5px solid #EA580C; color: #C2410C; }
    .adv-abort { background-color: #FEE2E2; border-left: 5px solid #DC2626; color: #B91C1C; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Top-Level Tab Navigation
# -----------------------------------------------------------------------------
tab_main, tab_fd001 = st.tabs([
    "✈️ AeroVigil DT — Aero-Engine Digital Twin",
    "🔬 FD001 PHM Validation (NASA C-MAPSS)",
])

# =============================================================================
# TAB 1: EXISTING AEROVIGIL DT DASHBOARD (UNCHANGED)
# =============================================================================
with tab_main:

 # ---------------------------------------------------------------------------
 # Sidebar Controls & Parameters (inside Tab 1 context)
 # ---------------------------------------------------------------------------
 st.sidebar.image("https://img.icons8.com/color/96/jet-engine.png", width=64)
 st.sidebar.title("AeroVigil DT Controls")
 st.sidebar.caption("SIH 2026 Problem Statement SIH26054")

 scenario_choice = st.sidebar.selectbox(
     "Select Telemetry Scenario",
     options=list(SCENARIO_DISPLAY_NAMES.values()),
     index=1,  # Default to Overheating Thermal Fault
     help="Select simulated flight mission scenario for testing.",
 )
 selected_scenario_key = SCENARIO_NAME_MAP[scenario_choice]

 duration_mins = st.sidebar.slider(
     "Mission Duration (Minutes)",
     min_value=5,
     max_value=20,
     value=20,
     step=5,
     help="Total simulated flight duration.",
 )

 random_seed = st.sidebar.number_input(
     "Deterministic Seed",
     min_value=1,
     max_value=9999,
     value=42,
     help="Random seed for exact reproducibility.",
 )

 st.sidebar.markdown("---")
 run_btn = st.sidebar.button("▶ Run Simulation", use_container_width=True, type="primary")

 # Execute / fetch cached simulation result
 df_processed = run_cached_simulation(
     scenario=selected_scenario_key,
     duration_minutes=float(duration_mins),
     seed=int(random_seed),
 )

 total_rows = len(df_processed)
 total_sec = float(df_processed["timestamp_sec"].iloc[-1])

 st.sidebar.markdown("---")
 st.sidebar.subheader("Mission Replay Controls")
 replay_step = st.sidebar.slider(
     "Replay Timestep (seconds)",
     min_value=0,
     max_value=int(total_sec),
     value=int(total_sec),
     step=1,
     help="Slide to inspect telemetry state at any point in the flight mission.",
 )

 st.sidebar.markdown("---")
 st.sidebar.info(
     "**Synthetic Prototype Disclaimer**:\n"
     "All telemetry, Digital Twin estimates, Health Indices, RUL projections, "
     "and risk advisories are synthetic prototype representations created for SIH 2026 demonstration."
 )

 # Extract snapshot at selected replay timestep
 snapshot = extract_snapshot_at_index(df_processed, replay_step)

 # ---------------------------------------------------------------------------
 # 3. Main Dashboard Header
 # ---------------------------------------------------------------------------
 st.markdown("<div class='main-header'>AeroVigil DT — Real-Time Aero-Engine Digital Twin</div>", unsafe_allow_html=True)
 st.markdown(
     f"<div class='sub-header'>Active Scenario: <b>{scenario_choice}</b> | Timestep: <b>{snapshot['timestamp_sec']:.0f}s / {total_sec:.0f}s</b> | Flight Phase: <b>{snapshot['mission_phase']}</b></div>",
     unsafe_allow_html=True,
 )

 # ---------------------------------------------------------------------------
 # 4. Top-Level KPI Cards
 # ---------------------------------------------------------------------------
 kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

 with kpi_col1:
     h_val = snapshot["health_index"]
     h_state = snapshot["health_state"]
     badge_cls = "badge-normal" if h_val >= 90 else "badge-warning" if h_val >= 75 else "badge-anomalous" if h_val >= 50 else "badge-critical"
     st.markdown(
         f"""
         <div class='kpi-card'>
             <div class='kpi-title'>Engine Health Index</div>
             <div class='kpi-value'>{h_val:.1f} <span style='font-size:14px;color:#64748B;'>/ 100</span></div>
             <div class='kpi-badge {badge_cls}'>{h_state} ({snapshot['health_trend']})</div>
         </div>
         """,
         unsafe_allow_html=True,
     )

 with kpi_col2:
     rul_disp = snapshot["rul_display"]
     rul_stat = snapshot["rul_status"]
     badge_cls = "badge-normal" if rul_stat == "STABLE" else "badge-warning" if rul_stat == "DEGRADING" else "badge-critical"
     st.markdown(
         f"""
         <div class='kpi-card'>
             <div class='kpi-title'>Prototype Est. RUL</div>
             <div class='kpi-value'>{rul_disp}</div>
             <div class='kpi-badge {badge_cls}'>{rul_stat}</div>
         </div>
         """,
         unsafe_allow_html=True,
     )

 with kpi_col3:
     r_score = snapshot["mission_risk_score"]
     r_level = snapshot["mission_risk_level"]
     badge_cls = "badge-normal" if r_level == "LOW" else "badge-warning" if r_level == "MODERATE" else "badge-anomalous" if r_level == "HIGH" else "badge-critical"
     st.markdown(
         f"""
         <div class='kpi-card'>
             <div class='kpi-title'>Mission Risk Score</div>
             <div class='kpi-value'>{r_score:.1f} <span style='font-size:14px;color:#64748B;'>/ 100</span></div>
             <div class='kpi-badge {badge_cls}'>RISK: {r_level}</div>
         </div>
         """,
         unsafe_allow_html=True,
     )

 with kpi_col4:
     flt_type = snapshot["fault_type"]
     sev = snapshot["severity"]
     badge_cls = "badge-normal" if flt_type == "NORMAL" else "badge-critical"
     st.markdown(
         f"""
         <div class='kpi-card'>
             <div class='kpi-title'>Classified Fault</div>
             <div class='kpi-value' style='font-size:18px;margin:8px 0;'>{flt_type}</div>
             <div class='kpi-badge {badge_cls}'>SEVERITY: {sev}</div>
         </div>
         """,
         unsafe_allow_html=True,
     )

 st.markdown("<br>", unsafe_allow_html=True)

 # ---------------------------------------------------------------------------
 # 5. Operational Advisory Banner
 # ---------------------------------------------------------------------------
 rec = snapshot["mission_recommendation"]
 adv_class = (
     "adv-continue" if rec == "CONTINUE_MONITORING"
     else "adv-increase" if rec == "INCREASE_MONITORING"
     else "adv-inspect" if rec == "INSPECT_AT_NEXT_OPPORTUNITY"
     else "adv-abort"
 )

 st.markdown(
     f"""
     <div class='advisory-box {adv_class}'>
         PROTOTYPE ADVISORY: {rec.replace('_', ' ')} &nbsp;|&nbsp;
         Phase: {snapshot['mission_phase']} &nbsp;|&nbsp;
         Anomaly Score: {snapshot['anomaly_score']:.1f} ({snapshot['anomaly_level']})
     </div>
     """,
     unsafe_allow_html=True,
 )

 st.markdown("<br>", unsafe_allow_html=True)

 # ---------------------------------------------------------------------------
 # 5.5 Interactive 3D Digital Twin
 # ---------------------------------------------------------------------------
 st.subheader("Interactive 3D Digital Twin")
 st.caption("Real-time visual representation of engine state, fault localization, and operating conditions.")
 
 col_3d, col_info = st.columns([3, 1])
 
 with col_3d:
     from src.engine_3d.component import engine_3d_component
     engine_3d_component(snapshot=snapshot, key="engine_viewer")
     
 with col_info:
     st.markdown("#### Engine State")
     st.write(f"**Health Index**: {snapshot['health_index']:.1f}")
     st.write(f"**RUL**: {snapshot['rul_display']}")
     st.write(f"**RPM**: {snapshot['RPM']:.0f}")
     st.write(f"**CHT**: {snapshot['CHT']:.1f} °C")
     st.write(f"**EGT**: {snapshot['EGT']:.1f} °C")
     st.write(f"**Oil Press**: {snapshot['oil_pressure']:.1f} psi")
     st.write(f"**Vibration**: {snapshot['vibration']:.3f} g")
     st.markdown("---")
     st.write(f"**Status**: `{snapshot['health_state']}`")
     st.write(f"**Fault**: `{snapshot['fault_type']}`")
     st.write(f"**Phase**: `{snapshot['mission_phase']}`")

 st.markdown("<br>", unsafe_allow_html=True)

 # ---------------------------------------------------------------------------
 # 6. Live Telemetry Panel
 # ---------------------------------------------------------------------------
 st.subheader("1. Live Engine Sensor Telemetry")
 m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
 m_col1.metric("RPM", f"{snapshot['RPM']:.1f} rev/min", delta=f"{snapshot['RPM'] - snapshot['expected_RPM']:.1f}")
 m_col2.metric("CHT", f"{snapshot['CHT']:.2f} °C", delta=f"{snapshot['residual_CHT']:+.2f} °C")
 m_col3.metric("EGT", f"{snapshot['EGT']:.2f} °C", delta=f"{snapshot['residual_EGT']:+.2f} °C")
 m_col4.metric("Oil Pressure", f"{snapshot['oil_pressure']:.2f} psi", delta=f"{snapshot['residual_oil_pressure']:+.2f} psi")
 m_col5.metric("Oil Temperature", f"{snapshot['oil_temperature']:.2f} °C")

 m_col6, m_col7, m_col8, m_col9, m_col10 = st.columns(5)
 m_col6.metric("Fuel Flow", f"{snapshot['fuel_flow']:.2f} L/h")
 m_col7.metric("Vibration", f"{snapshot['vibration']:.4f} g")
 m_col8.metric("Battery Voltage", f"{snapshot['battery_voltage']:.2f} V")
 m_col9.metric("Throttle Command", f"{snapshot['throttle']*100:.1f} %")
 m_col10.metric("Ambient Temp", f"{snapshot['ambient_temperature']:.1f} °C")

 st.markdown("<br>", unsafe_allow_html=True)

 # ---------------------------------------------------------------------------
 # 7. Digital Twin Visualization Panel (Observed vs Expected)
 # ---------------------------------------------------------------------------
 st.subheader("2. Digital Twin: Observed Telemetry vs Expected Physics State")
 st.caption("Central Technical Differentiator: Comparing observed sensor values against the Digital Twin's physics-informed surrogate predictions.")

 dt_col1, dt_col2, dt_col3 = st.columns(3)

 t_sec = df_processed["timestamp_sec"]

 with dt_col1:
     fig_cht, ax_cht = plt.subplots(figsize=(5, 3.5))
     ax_cht.plot(t_sec, df_processed["CHT"], label="Observed CHT", color="#d62728", linewidth=1.5)
     ax_cht.plot(t_sec, df_processed["expected_CHT"], label="Expected CHT (Digital Twin)", color="#1f77b4", linestyle="--", linewidth=1.5)
     ax_cht.axvline(snapshot["timestamp_sec"], color="black", linestyle=":", label="Replay Time")
     ax_cht.set_ylabel("CHT (°C)")
     ax_cht.set_xlabel("Elapsed Time (s)")
     ax_cht.set_title("Cylinder Head Temp (CHT)")
     ax_cht.legend(loc="upper left", fontsize=8)
     ax_cht.grid(True, linestyle="--", alpha=0.5)
     st.pyplot(fig_cht)

 with dt_col2:
     fig_egt, ax_egt = plt.subplots(figsize=(5, 3.5))
     ax_egt.plot(t_sec, df_processed["EGT"], label="Observed EGT", color="#e377c2", linewidth=1.5)
     ax_egt.plot(t_sec, df_processed["expected_EGT"], label="Expected EGT (Digital Twin)", color="#1f77b4", linestyle="--", linewidth=1.5)
     ax_egt.axvline(snapshot["timestamp_sec"], color="black", linestyle=":", label="Replay Time")
     ax_egt.set_ylabel("EGT (°C)")
     ax_egt.set_xlabel("Elapsed Time (s)")
     ax_egt.set_title("Exhaust Gas Temp (EGT)")
     ax_egt.legend(loc="upper left", fontsize=8)
     ax_egt.grid(True, linestyle="--", alpha=0.5)
     st.pyplot(fig_egt)

 with dt_col3:
     fig_oil, ax_oil = plt.subplots(figsize=(5, 3.5))
     ax_oil.plot(t_sec, df_processed["oil_pressure"], label="Observed Oil Press", color="#2ca02c", linewidth=1.5)
     ax_oil.plot(t_sec, df_processed["expected_oil_pressure"], label="Expected Oil Press", color="#1f77b4", linestyle="--", linewidth=1.5)
     ax_oil.axvline(snapshot["timestamp_sec"], color="black", linestyle=":", label="Replay Time")
     ax_oil.set_ylabel("Oil Pressure (psi)")
     ax_oil.set_xlabel("Elapsed Time (s)")
     ax_oil.set_title("Lubrication Oil Pressure")
     ax_oil.legend(loc="lower left", fontsize=8)
     ax_oil.grid(True, linestyle="--", alpha=0.5)
     st.pyplot(fig_oil)

 st.markdown("<br>", unsafe_allow_html=True)

 # ---------------------------------------------------------------------------
 # 8. Residual & Anomaly Analysis Panel
 # ---------------------------------------------------------------------------
 st.subheader("3. Residual Analysis & Anomaly Detection")
 res_col1, res_col2 = st.columns([2, 1])

 with res_col1:
     fig_res, ax_res = plt.subplots(figsize=(8, 3.5))
     ax_res.plot(t_sec, df_processed["normalized_residual_CHT"], label="CHT Norm Residual (σ)", color="#d62728", linewidth=1.2)
     ax_res.plot(t_sec, df_processed["normalized_residual_EGT"], label="EGT Norm Residual (σ)", color="#e377c2", linewidth=1.2)
     ax_res.plot(t_sec, df_processed["normalized_residual_oil_pressure"], label="Oil Press Norm Residual (σ)", color="#2ca02c", linewidth=1.2)
     ax_res.plot(t_sec, df_processed["normalized_residual_vibration"], label="Vibration Norm Residual (σ)", color="#9467bd", linewidth=1.2)
     ax_res.axhline(3.0, color="red", linestyle="--", alpha=0.7, label="3σ Anomaly Threshold")
     ax_res.axhline(-3.0, color="red", linestyle="--", alpha=0.7)
     ax_res.axvline(snapshot["timestamp_sec"], color="black", linestyle=":")
     ax_res.set_ylabel("Normalized Residual (σ)")
     ax_res.set_xlabel("Elapsed Time (s)")
     ax_res.set_title("Multi-Signal Normalized Residual Signatures (z-score)")
     ax_res.legend(loc="upper left", fontsize=8)
     ax_res.grid(True, linestyle="--", alpha=0.5)
     st.pyplot(fig_res)

 with res_col2:
     st.markdown("#### Diagnostic Explanation")
     st.write(f"**Classified Fault**: `{snapshot['fault_type']}`")
     st.write(f"**Severity**: `{snapshot['severity']}`")
     st.write(f"**Evidence Score**: `{snapshot['evidence_display']}`")
     st.write(f"**Contributing Signals**: `{snapshot['contributing_signals'] if snapshot['contributing_signals'] else 'None'}`")
     st.markdown(f"**Reasoning**: *{snapshot['reasoning']}*")

 st.markdown("<br>", unsafe_allow_html=True)

 # ---------------------------------------------------------------------------
 # 9. Health & RUL Trajectories
 # ---------------------------------------------------------------------------
 st.subheader("4. Engine Health Index & Prototype RUL Trajectories")
 h_col1, h_col2 = st.columns(2)

 with h_col1:
     fig_h, ax_h = plt.subplots(figsize=(6, 3.5))
     ax_h.plot(t_sec, df_processed["health_index"], color="#1f77b4", linewidth=1.8, label="Health Index")
     ax_h.axhline(90.0, color="green", linestyle="--", alpha=0.5, label="Healthy (90)")
     ax_h.axhline(75.0, color="orange", linestyle="--", alpha=0.5, label="Degraded (75)")
     ax_h.axhline(50.0, color="darkorange", linestyle="--", alpha=0.5, label="Warning (50)")
     ax_h.axhline(25.0, color="red", linestyle=":", alpha=0.7, label="Critical (25)")
     ax_h.axvline(snapshot["timestamp_sec"], color="black", linestyle=":")
     ax_h.set_ylabel("Health Index (0-100)")
     ax_h.set_xlabel("Elapsed Time (s)")
     ax_h.set_title("Engine Health Index Trajectory")
     ax_h.legend(loc="lower left", fontsize=8)
     ax_h.grid(True, linestyle="--", alpha=0.5)
     st.pyplot(fig_h)

 with h_col2:
     fig_rul, ax_rul = plt.subplots(figsize=(6, 3.5))
     valid_rul = df_processed["rul_minutes"]
     ax_rul.plot(t_sec, valid_rul, color="#ff7f0e", linewidth=1.8, label="Est. Prototype RUL (min)")
     ax_rul.axhline(10.0, color="orange", linestyle="--", alpha=0.5, label="Low RUL (10 min)")
     ax_rul.axhline(3.0, color="red", linestyle=":", alpha=0.7, label="Critical RUL (3 min)")
     ax_rul.axvline(snapshot["timestamp_sec"], color="black", linestyle=":")
     ax_rul.set_ylabel("RUL (minutes)")
     ax_rul.set_xlabel("Elapsed Time (s)")
     ax_rul.set_title("Prototype Remaining Useful Life (RUL)")
     ax_rul.legend(loc="upper right", fontsize=8)
     ax_rul.grid(True, linestyle="--", alpha=0.5)
     st.pyplot(fig_rul)
     st.caption("Note: RUL is projected from recent health degradation slope (dH/dt).")

 st.markdown("<br>", unsafe_allow_html=True)

 # ---------------------------------------------------------------------------
 # 10. Mission Risk Timeline & Scenario Comparison Matrix
 # ---------------------------------------------------------------------------
 st.subheader("5. Mission Risk Assessment & Cross-Scenario Comparison")

 risk_tab, comp_tab = st.tabs(["Mission Risk Timeline", "Cross-Scenario Evaluation Matrix"])

 with risk_tab:
     fig_r, ax_r = plt.subplots(figsize=(10, 3.5))
     ax_r.plot(t_sec, df_processed["mission_risk_score"], color="#d62728", linewidth=1.8, label="Mission Risk Score")
     ax_r.axhline(20.0, color="green", linestyle="--", alpha=0.5, label="Low Risk (<20)")
     ax_r.axhline(45.0, color="orange", linestyle="--", alpha=0.5, label="Moderate Risk (<45)")
     ax_r.axhline(70.0, color="red", linestyle=":", alpha=0.7, label="High Risk (<70)")
     ax_r.axvline(snapshot["timestamp_sec"], color="black", linestyle=":")
     ax_r.set_ylabel("Risk Score (0-100)")
     ax_r.set_xlabel("Elapsed Time (s)")
     ax_r.set_title("Flight-Phase Aware Mission Risk Score")
     ax_r.legend(loc="upper left", fontsize=8)
     ax_r.grid(True, linestyle="--", alpha=0.5)
     st.pyplot(fig_r)

 with comp_tab:
     st.markdown("#### Scenario Evaluation Comparison Matrix")
     st.caption("Executes all 5 telemetry scenarios to demonstrate distinct diagnostic signatures and risk profiles.")

     comp_df = generate_scenario_comparison(duration_minutes=float(duration_mins), seed=int(random_seed))
     st.dataframe(comp_df.drop(columns=["Scenario Key"]), use_container_width=True, hide_index=True)


# =============================================================================
# TAB 2: FD001 PHM VALIDATION DASHBOARD
# =============================================================================
with tab_fd001:
    st.markdown("<div class='main-header'>🔬 NASA C-MAPSS FD001 — PHM Validation Dashboard</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Data-Driven PHM Validation · Degradation Modelling · "
        "Health Index · ML-Based RUL Prediction · Fleet Evaluation</div>",
        unsafe_allow_html=True,
    )

    # Domain disclaimer banner
    st.info(
        "⚠️ **Domain Disclaimer**: This tab uses the NASA C-MAPSS FD001 dataset "
        "(turbofan HPC degradation) to validate **generic PHM capabilities** of AeroVigil. "
        "C-MAPSS FD001 does **NOT** represent an aero-piston engine. Results demonstrate "
        "degradation modelling, health index computation, and RUL prediction accuracy, "
        "but do **NOT** directly calibrate or replace the AeroVigil piston-engine physics "
        "model or UAV mission-reliability layer."
    )

    if not _FD001_AVAILABLE:
        st.warning(
            "FD001 trained artifacts not found. "
            "Please run the training script first:\n\n"
            "```bash\npython scripts/train_fd001_rul.py\n```"
        )
        st.stop()

    # -----------------------------------------------------------------------
    # Load pre-computed artifacts
    # -----------------------------------------------------------------------
    @st.cache_data
    def _load_fd001_artifacts():
        """Loads pre-trained FD001 artifacts from disk."""
        with open(_FD001_ARTIFACTS_DIR / "gbr_rul_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(_FD001_ARTIFACTS_DIR / "preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        preds_df = pd.read_csv(_FD001_ARTIFACTS_DIR / "test_predictions.csv")
        health_df = pd.read_csv(_FD001_ARTIFACTS_DIR / "sample_health_trajectories.csv")
        with open(_FD001_ARTIFACTS_DIR / "metrics_summary.json") as f:
            metrics = json.load(f)
        return model, preprocessor, preds_df, health_df, metrics

    fd_model, fd_prep, fd_preds, fd_health, fd_metrics = _load_fd001_artifacts()

    # -----------------------------------------------------------------------
    # Metrics KPIs
    # -----------------------------------------------------------------------
    st.subheader("1. Model Validation Metrics (100 Test Engines vs RUL_FD001.txt)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test MAE", f"{fd_metrics['test_mae']:.2f} cycles",
              help="Mean Absolute Error on 100 test engines")
    m2.metric("Test RMSE", f"{fd_metrics['test_rmse']:.2f} cycles",
              help="Root Mean Squared Error")
    m3.metric("Test R²", f"{fd_metrics['test_r2']:.3f}",
              help="Coefficient of determination")
    m4.metric("NASA Score", f"{fd_metrics['test_nasa_score']:.1f}",
              help="NASA asymmetric PHM score (lower = better). Penalises late predictions more than early.")

    st.caption(
        f"Model: Gradient Boosting Regressor · "
        f"Features: {fd_metrics['n_sensors_retained']} sensors × (mean+std) over 15-cycle window · "
        f"Engine split: {fd_metrics['n_engines_train']} train / {fd_metrics['n_engines_val']} val · "
        f"RUL cap: {fd_metrics['rul_cap']} cycles"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Sub-tabs: Per-Engine View | Fleet Validation | Sensor Analysis
    # -----------------------------------------------------------------------
    eng_tab, fleet_tab, sensor_tab = st.tabs([
        "🔎 Per-Engine Inspector",
        "📊 Fleet Validation",
        "📡 Sensor / Health Trajectories",
    ])

    # --- Per-Engine Inspector ---
    with eng_tab:
        st.subheader("Individual Test Engine Inspector")
        engine_ids = sorted(fd_preds["unit_number"].tolist())
        sel_eng = st.selectbox("Select Test Engine ID", engine_ids, key="fd001_engine_select")

        eng_row = fd_preds[fd_preds["unit_number"] == sel_eng].iloc[0]
        pred_rul = float(eng_row["pred_rul"])
        true_rul = float(eng_row["true_rul"])
        error = float(eng_row["error"])
        abs_error = float(eng_row["abs_error"])

        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Engine ID", f"#{int(sel_eng)}")
        ec2.metric("Predicted RUL", f"{pred_rul:.0f} cycles")
        ec3.metric("True RUL (Ground Truth)", f"{true_rul:.0f} cycles",
                   delta=f"{error:+.0f} cycles",
                   delta_color="inverse")
        err_pct = (abs_error / max(1.0, true_rul)) * 100.0
        ec4.metric("Absolute Error", f"{abs_error:.0f} cycles",
                   help=f"{err_pct:.1f}% of true RUL")

        # Health trajectory for this engine (from training data proxy)
        if sel_eng in fd_health["unit_number"].values:
            eng_h = fd_health[fd_health["unit_number"] == sel_eng].sort_values("cycle")
            hc1, hc2 = st.columns(2)
            with hc1:
                fig_eh, ax_eh = plt.subplots(figsize=(6, 3.5))
                ax_eh.plot(eng_h["cycle"], eng_h["health_index"],
                           color="#1f77b4", linewidth=1.8)
                ax_eh.axhline(75.0, color="orange", linestyle="--", alpha=0.6)
                ax_eh.axhline(50.0, color="red", linestyle=":", alpha=0.6)
                ax_eh.set_xlabel("Operational Cycle")
                ax_eh.set_ylabel("Model-Derived Health Index (0–100)")
                ax_eh.set_title(f"Engine {int(sel_eng)} — Health Trajectory\n"
                                "(Model-Derived, data-driven, not physically measured)")
                ax_eh.grid(True, alpha=0.4)
                st.pyplot(fig_eh)
            with hc2:
                fig_es, ax_es = plt.subplots(figsize=(6, 3.5))
                ax_es.plot(eng_h["cycle"], eng_h["anomaly_score"],
                           color="#d62728", linewidth=1.5)
                ax_es.axhline(35.0, color="orange", linestyle="--", alpha=0.6, label="Warning (35)")
                ax_es.axhline(65.0, color="red", linestyle=":", alpha=0.6, label="Anomalous (65)")
                ax_es.set_xlabel("Operational Cycle")
                ax_es.set_ylabel("Anomaly Score (0–100)")
                ax_es.set_title(f"Engine {int(sel_eng)} — Degradation Anomaly Score\n"
                                "(Data-Driven Residual from Healthy Baseline)")
                ax_es.legend(fontsize=8)
                ax_es.grid(True, alpha=0.4)
                st.pyplot(fig_es)
        else:
            st.info("Health trajectory preview is available only for training engines (first 10).")

    # --- Fleet Validation ---
    with fleet_tab:
        st.subheader("Fleet-Level RUL Validation (100 Test Engines)")
        fl1, fl2 = st.columns(2)

        with fl1:
            fig_fl1, ax_fl1 = plt.subplots(figsize=(6, 5))
            true_arr = fd_preds["true_rul"].values
            pred_arr = fd_preds["pred_rul"].values
            ax_fl1.scatter(true_arr, pred_arr, alpha=0.6, color="#1f77b4", s=35)
            mx = max(true_arr.max(), pred_arr.max()) + 10
            ax_fl1.plot([0, mx], [0, mx], "r--", lw=1.5, label="Perfect Prediction")
            ax_fl1.set_xlabel("True RUL (cycles)")
            ax_fl1.set_ylabel("Predicted RUL (cycles)")
            ax_fl1.set_title("Predicted vs True RUL\n(C-MAPSS FD001, NOT aero-piston engine)")
            ax_fl1.text(0.05, 0.92,
                        f"MAE={fd_metrics['test_mae']:.1f}  RMSE={fd_metrics['test_rmse']:.1f}  R²={fd_metrics['test_r2']:.3f}",
                        transform=ax_fl1.transAxes, fontsize=9,
                        bbox=dict(facecolor="lightyellow", alpha=0.8, edgecolor="gray"))
            ax_fl1.legend(fontsize=8)
            ax_fl1.grid(True, alpha=0.4)
            st.pyplot(fig_fl1)

        with fl2:
            fig_fl2, ax_fl2 = plt.subplots(figsize=(6, 5))
            errors = fd_preds["error"].values
            ax_fl2.hist(errors, bins=20, color="#ff7f0e", edgecolor="black", alpha=0.75)
            ax_fl2.axvline(0, color="red", linestyle="--", label="Zero error")
            ax_fl2.axvline(errors.mean(), color="blue", linestyle=":",
                           label=f"Mean={errors.mean():.1f}")
            ax_fl2.set_xlabel("Error = Predicted − True (cycles)")
            ax_fl2.set_ylabel("Count")
            ax_fl2.set_title("RUL Prediction Error Distribution\n(FD001 Test Engines)")
            ax_fl2.legend(fontsize=8)
            ax_fl2.grid(True, alpha=0.4)
            st.pyplot(fig_fl2)

        st.subheader("Per-Engine Prediction Table (First 20 Engines)")
        disp_df = fd_preds[["unit_number", "pred_rul", "true_rul", "error", "abs_error"]].copy()
        disp_df.columns = ["Engine ID", "Pred. RUL (cycles)", "True RUL (cycles)",
                           "Error (cycles)", "Abs Error (cycles)"]
        disp_df["Pred. RUL (cycles)"] = disp_df["Pred. RUL (cycles)"].round(1)
        disp_df["Error (cycles)"] = disp_df["Error (cycles)"].round(1)
        disp_df["Abs Error (cycles)"] = disp_df["Abs Error (cycles)"].round(1)
        st.dataframe(disp_df.head(20), use_container_width=True, hide_index=True)

    # --- Sensor / Health Trajectories ---
    with sensor_tab:
        st.subheader("Health Trajectories — Sample Training Engines")
        st.caption(
            "Showing model-derived health indices for 10 sample training engines. "
            "Each trajectory starts near 100 (healthy) and degrades monotonically "
            "toward failure. Computed from data-driven baseline deviation, not "
            "from the physics-based piston-engine Digital Twin."
        )
        fig_ht, ax_ht = plt.subplots(figsize=(12, 5))
        colors_ht = plt.cm.tab10.colors
        for i, eng_id in enumerate(sorted(fd_health["unit_number"].unique())[:8]):
            eng_h = fd_health[fd_health["unit_number"] == eng_id]
            ax_ht.plot(eng_h["cycle"], eng_h["health_index"],
                       color=colors_ht[i % 10], linewidth=1.2,
                       label=f"Engine {int(eng_id)}")
        ax_ht.axhline(75.0, color="orange", linestyle="--", alpha=0.6, label="Degraded (75)")
        ax_ht.axhline(50.0, color="red", linestyle=":", alpha=0.6, label="Warning (50)")
        ax_ht.set_xlabel("Operational Cycle")
        ax_ht.set_ylabel("Model-Derived Health Index (0–100)")
        ax_ht.set_title("FD001 Engine Degradation Trajectories\n"
                        "(Model-Derived Health Index — data-driven, not physically measured)")
        ax_ht.legend(ncol=4, fontsize=8)
        ax_ht.grid(True, alpha=0.4)
        st.pyplot(fig_ht)

        st.markdown("---")
        st.subheader("Sensor Selection Summary")
        sensor_cols = st.columns(2)
        with sensor_cols[0]:
            st.markdown("**Retained Sensors** (informative, used in RUL model):")
            for s in fd_metrics["sensors_retained"]:
                st.markdown(f"  - `{s}`")
        with sensor_cols[1]:
            st.markdown("**Dropped Sensors** (constant / near-constant, std < 0.001):")
            for s in fd_metrics["sensors_dropped"]:
                st.markdown(f"  - `{s}` — near-constant across all engines")

    # -----------------------------------------------------------------------
    # Domain boundary reminder
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### ⚠️ Legitimate SIH Validation Claims (What Can Be Claimed)")
    st.markdown(f"""
    | Claim | Supported |
    |---|---|
    | AeroVigil PHM pipeline achieves MAE ≈ **{fd_metrics['test_mae']:.1f} cycles** on FD001 benchmark | ✅ Yes |
    | RMSE ≈ **{fd_metrics['test_rmse']:.1f} cycles**, R² ≈ **{fd_metrics['test_r2']:.3f}** on 100 test engines | ✅ Yes |
    | Data-driven health index monotonically captures HPC degradation | ✅ Yes |
    | No data leakage: engines split 80/20 by ID; test GT never seen in training | ✅ Yes |
    | C-MAPSS FD001 validates aero-**piston** engine physics | ❌ No — turbofan ≠ piston |
    | FD001 calibrates the AeroVigil Digital Twin physics model | ❌ No — separate models |
    | FD001 validates UAV mission risk values | ❌ No — mission model is separate |
    """)
