"""
Classification module for NTRO Thermal Detection and Classification system (SIH26162).
"""

from src.classification.heuristic_baseline import (
    ThermalAnomalyClass,
    CLASS_NAMES,
    CLASS_SHORT_LABELS,
    HeuristicBaselineClassifier,
    heuristic_baseline_classify,
)
from src.classification.risk_scorer import (
    RiskScorer,
    AlertLevel,
    calculate_risk_severity_index,
)
from src.classification.ensemble_classifier import (
    PredictionOutput,
    ThermalEnsembleClassifier,
)
from src.classification.model_trainer import ModelTrainer

__all__ = [
    "ThermalAnomalyClass",
    "CLASS_NAMES",
    "CLASS_SHORT_LABELS",
    "HeuristicBaselineClassifier",
    "heuristic_baseline_classify",
    "RiskScorer",
    "AlertLevel",
    "calculate_risk_severity_index",
    "PredictionOutput",
    "ThermalEnsembleClassifier",
    "ModelTrainer",
]
