"""
AeroVigil - Data Ingestion Package

Provides clean dataset adapters for external benchmark datasets.
Currently supports NASA C-MAPSS FD001 as a PHM validation data source.

Note: This package is intentionally decoupled from the core AeroVigil
aero-piston-engine physics model. It validates generic PHM capabilities
(degradation modelling, anomaly detection, RUL prediction) using
publicly available benchmark data.
"""

from .cmapss_loader import CMAPSSLoader, load_fd001
from .cmapss_preprocessor import CMAPSSPreprocessor
from .cmapss_adapter import CMAPSSResidualAdapter
from .cmapss_rul_model import CMAPSSRULModel

__all__ = [
    "CMAPSSLoader",
    "load_fd001",
    "CMAPSSPreprocessor",
    "CMAPSSResidualAdapter",
    "CMAPSSRULModel",
]
