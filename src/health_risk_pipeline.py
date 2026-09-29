"""
AeroVigil DT - Integrated Pipeline Component
Phase 4 Integration Layer

Chains Telemetry -> Digital Twin -> Residual Analyzer -> Health Index -> RUL Estimator -> Mission Risk Model
into a unified end-to-end processing pipeline.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

import sys
import os

# Ensure repo root is on path if running as standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.simulator import generate_telemetry
from src.digital_twin import DigitalTwin
from src.residual_analysis import ResidualAnalyzer
from src.health_index import EngineHealthIndex
from src.rul_estimator import RULEstimator
from src.mission_risk import MissionRiskEstimator


class AeroVigilPipeline:
    """
    Unified AeroVigil DT End-to-End Processing Pipeline.

    Orchestrates expected-state generation, residual analysis, fault classification,
    health index calculation, prototype RUL estimation, and mission risk assessment.
    """

    def __init__(self):
        self.digital_twin = DigitalTwin()
        self.residual_analyzer = ResidualAnalyzer()
        self.health_index_calc = EngineHealthIndex()
        self.rul_estimator = RULEstimator()
        self.mission_risk_estimator = MissionRiskEstimator()

    def process(self, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes end-to-end pipeline processing on input telemetry DataFrame.

        Parameters:
        -----------
        telemetry_df : pd.DataFrame
            Raw telemetry DataFrame.

        Returns:
        --------
        pd.DataFrame
            Augmented DataFrame containing expected states, residuals, fault classification,
            health index, RUL estimates, and mission risk advisories.
        """
        # Step 1: Digital Twin Expected-State & Residuals
        df_dt = self.digital_twin.predict_expected_state(telemetry_df)

        # Step 2: Residual Analysis, Anomaly Detection & Fault Classification
        df_diag = self.residual_analyzer.analyze(df_dt)

        # Step 3: Engine Health Index Calculation
        df_health = self.health_index_calc.calculate(df_diag)

        # Step 4: Prototype RUL Estimation
        df_rul = self.rul_estimator.estimate(df_health)

        # Step 5: Mission Risk Model & Advisory Assessment
        df_final = self.mission_risk_estimator.assess(df_rul)

        return df_final


def run_pipeline(telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """Convenience functional wrapper for executing the AeroVigil DT pipeline."""
    pipeline = AeroVigilPipeline()
    return pipeline.process(telemetry_df)
