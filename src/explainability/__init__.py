"""
Explainability (XAI) module for NTRO Thermal Detection and Classification system (SIH26162).
"""

from src.explainability.shap_engine import (
    SHAPExplanation,
    SHAPExplainerEngine,
)
from src.explainability.natural_language_xai import NaturalLanguageXAI

__all__ = [
    "SHAPExplanation",
    "SHAPExplainerEngine",
    "NaturalLanguageXAI",
]
