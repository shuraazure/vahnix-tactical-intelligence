"""
Deterministic Rule-Based Baseline Classifier for NTRO Thermal Detection System.
Provides transparent domain heuristic classifications for benchmarking ML models.
Authoritative Specifications: ORIGINAL_REQUEST.md § Acceptance Criteria, PROJECT.md § 2.2
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Sequence, Union
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, accuracy_score


class ThermalAnomalyClass(Enum):
    CONTROLLED_INDUSTRIAL_FLARE = 0
    INDUSTRIAL_FIRE_DISASTER = 1
    AGRICULTURAL_STUBBLE_BURNING = 2
    WILDFIRE_VEGETATION_BURNING = 3
    VOLCANIC_ACTIVITY = 4
    NOISE = 5


CLASS_NAMES: dict[int, str] = {
    0: "Controlled Industrial Flare",
    1: "Industrial Fire Emergency",
    2: "Agricultural Burning",
    3: "Wildfire",
    4: "Volcanic Activity",
    5: "Noise",
}

CLASS_SHORT_LABELS: dict[int, str] = {
    0: "FLARE",
    1: "INDUSTRIAL_EMERGENCY",
    2: "AGRICULTURAL",
    3: "WILDFIRE",
    4: "VOLCANIC",
    5: "NOISE",
}


def heuristic_baseline_classify(features: Union[np.ndarray, Dict[str, float]]) -> ThermalAnomalyClass:
    """
    Deterministic domain rule engine for thermal anomaly classification.
    Feature indices match FEATURE_NAMES_28D:
      - 0: frp
      - 3: brightness_delta
      - 4: frp_zscore_cluster
      - 9: temporal_span_days
      - 10: recurrence_freq_per_month
      - 11: day_night_ratio
      - 14: cluster_radius_m
      - 17: dist_to_industrial_m
      - 18: is_inside_industrial
    """
    if isinstance(features, np.ndarray):
        dist_ind = float(features[17]) if len(features) > 17 else 5000.0
        is_inside = float(features[18]) if len(features) > 18 else 0.0
        frp = float(features[0]) if len(features) > 0 else 20.0
        frp_zscore = float(features[4]) if len(features) > 4 else 0.0
        delta_t = float(features[3]) if len(features) > 3 else 30.0
        recur_freq = float(features[10]) if len(features) > 10 else 1.0
        t_span = float(features[9]) if len(features) > 9 else 1.0
        dn_ratio = float(features[11]) if len(features) > 11 else 1.0
        c_radius = float(features[14]) if len(features) > 14 else 100.0
    elif isinstance(features, dict):
        dist_ind = float(features.get("dist_to_industrial_m", 5000.0))
        is_inside = float(features.get("is_inside_industrial", 0.0))
        frp = float(features.get("frp", 20.0))
        frp_zscore = float(features.get("frp_zscore_cluster", 0.0))
        delta_t = float(features.get("brightness_delta", 30.0))
        recur_freq = float(features.get("recurrence_freq_per_month", 1.0))
        t_span = float(features.get("temporal_span_days", 1.0))
        dn_ratio = float(features.get("day_night_ratio", 1.0))
        c_radius = float(features.get("cluster_radius_m", 100.0))
    else:
        return ThermalAnomalyClass.WILDFIRE_VEGETATION_BURNING

    # Rule 1: Industrial Fire Emergency
    # High FRP or high delta-T or extreme Z-score inside or near industrial facilities
    if (dist_ind <= 500.0 or is_inside == 1.0) and (frp >= 150.0 or frp_zscore >= 3.0 or delta_t >= 60.0):
        return ThermalAnomalyClass.INDUSTRIAL_FIRE_DISASTER

    # Rule 2: Controlled Industrial Flare / Persistent Heat
    # Recurrent or long-spanning thermal anomalies strictly within/near industrial bounds
    if (dist_ind <= 350.0 or is_inside == 1.0) and (recur_freq >= 3.0 or t_span >= 5.0):
        return ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE

    # Rule 3: Agricultural Stubble Burning
    # Rural/cropland anomalies far from industry with strong daytime diurnal dominance and short lifespan
    if dist_ind > 1000.0 and dn_ratio >= 3.0 and t_span <= 4.0:
        return ThermalAnomalyClass.AGRICULTURAL_STUBBLE_BURNING

    # Rule 4: Wildfire / Vegetation
    # Expansive spatial radius or multi-day sustained front in non-industrial terrain
    if dist_ind > 1000.0 and (c_radius >= 1000.0 or t_span > 4.0 or frp >= 50.0):
        return ThermalAnomalyClass.WILDFIRE_VEGETATION_BURNING

    # Default fallback
    if dist_ind <= 750.0:
        return ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE
    return ThermalAnomalyClass.WILDFIRE_VEGETATION_BURNING


class HeuristicBaselineClassifier:
    """
    Scikit-learn compatible heuristic baseline classifier wrapper.
    """

    def __init__(self):
        self.classes_ = np.array([0, 1, 2, 3])

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> HeuristicBaselineClassifier:
        """No-op fit method for compatibility."""
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices [0, 1, 2, 3] for a matrix of features."""
        predictions = [heuristic_baseline_classify(row).value for row in X]
        return np.array(predictions, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return pseudo-probabilities (one-hot or smoothed) based on heuristic rules."""
        preds = self.predict(X)
        n = len(X)
        proba = np.full((n, 4), 0.05, dtype=float)
        for i, p in enumerate(preds):
            proba[i, p] = 0.85
        # Normalize
        row_sums = proba.sum(axis=1, keepdims=True)
        return proba / row_sums

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Evaluate baseline metrics against ground truth."""
        y_pred = self.predict(X_test)
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        accuracy = float(accuracy_score(y_test, y_pred))
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2, 3], zero_division=0)
        
        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "flare_f1": float(f1[0]),
            "emergency_recall": float(r[1]),
            "emergency_f1": float(f1[1]),
            "agricultural_f1": float(f1[2]),
            "wildfire_f1": float(f1[3])
        }
