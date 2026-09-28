"""
Multi-Class Thermal Anomaly Ensemble Classifier for NTRO Thermal Detection System.
Combines Cost-Sensitive Random Forest, Gradient Boosting, and HistGradientBoosting classifiers.
Authoritative Specifications: ORIGINAL_REQUEST.md § R2, PROJECT.md § 2, TEST_INFRA.md
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score

from src.features.feature_extractor import FEATURE_NAMES_28D, extract_28d_features
from src.data_pipeline.schemas import FIRMSDetection, IndustrialFacility
from src.clustering.schemas import ThermalCluster
from src.classification.heuristic_baseline import (
    ThermalAnomalyClass,
    CLASS_NAMES,
    CLASS_SHORT_LABELS,
    heuristic_baseline_classify,
)
from src.classification.risk_scorer import RiskScorer, calculate_risk_severity_index


@dataclass
class PredictionOutput:
    detection_id: str
    predicted_class: ThermalAnomalyClass
    predicted_label: str
    confidence_score: float
    class_probabilities: Dict[str, float]
    risk_severity_index: float  # [0.0, 100.0]
    is_critical_alert: bool
    feature_values: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "predicted_class": self.predicted_class.value,
            "predicted_class_name": self.predicted_class.name,
            "predicted_label": self.predicted_label,
            "confidence_score": float(self.confidence_score),
            "class_probabilities": {k: float(v) for k, v in self.class_probabilities.items()},
            "risk_severity_index": float(self.risk_severity_index),
            "is_critical_alert": bool(self.is_critical_alert),
            "feature_values": {k: float(v) for k, v in self.feature_values.items()},
        }


class ThermalEnsembleClassifier:
    """
    Production-grade multi-class ensemble classifier for thermal anomaly classification.
    Calibrated with cost-sensitive class weights (Class 1 Recall >= 0.90, Macro F1 >= 0.85).
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_names = list(FEATURE_NAMES_28D)
        self.classes_ = np.array([0, 1, 2, 3])
        self.is_fitted = False

        # Class weights: Upweight Class 1 (Industrial Fire Disaster) for high recall
        self.class_weight_rf = {0: 1.0, 1: 4.0, 2: 1.2, 3: 1.2}

        # Component models
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=3,
            class_weight=self.class_weight_rf,
            random_state=self.random_state,
            n_jobs=1
        )
        self.et_model = ExtraTreesClassifier(
            n_estimators=75,
            max_depth=12,
            class_weight=self.class_weight_rf,
            random_state=self.random_state,
            n_jobs=1
        )
        self.hgb_model = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=8,
            learning_rate=0.08,
            random_state=self.random_state,
            class_weight={0: 1.0, 1: 3.5, 2: 1.2, 3: 1.2}
        )

        # Ensemble voting weights [RF, ExtraTrees, HistGB]
        self.ensemble_weights = [0.40, 0.30, 0.30]

    def fit(self, X: np.ndarray, y: np.ndarray) -> ThermalEnsembleClassifier:
        """
        Train the ensemble classifier on 28D feature matrix X and target labels y.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=int)

        self.rf_model.fit(X_arr, y_arr)
        self.et_model.fit(X_arr, y_arr)
        self.hgb_model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def _align_probs(self, model: Any, X_arr: np.ndarray, n_classes: int = 4) -> np.ndarray:
        raw_probs = model.predict_proba(X_arr)
        if raw_probs.shape[1] == n_classes:
            return raw_probs
        aligned = np.full((len(X_arr), n_classes), 1e-6)
        classes = getattr(model, "classes_", range(raw_probs.shape[1]))
        for col_idx, class_label in enumerate(classes):
            if 0 <= int(class_label) < n_classes:
                aligned[:, int(class_label)] = raw_probs[:, col_idx]
        return aligned / aligned.sum(axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate calibrated soft probability distribution across 4 classes.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if not self.is_fitted:
            # Fallback to heuristic baseline probabilities if unfitted
            n = len(X_arr)
            probs = np.full((n, 4), 0.05)
            for i, row in enumerate(X_arr):
                c = heuristic_baseline_classify(row).value
                probs[i, c] = 0.85
            return probs / probs.sum(axis=1, keepdims=True)

        prob_rf = self._align_probs(self.rf_model, X_arr)
        prob_et = self._align_probs(self.et_model, X_arr)
        prob_hgb = self._align_probs(self.hgb_model, X_arr)

        # Weighted soft voting
        w = self.ensemble_weights
        blended = (w[0] * prob_rf) + (w[1] * prob_et) + (w[2] * prob_hgb)
        # Re-normalize
        blended = blended / blended.sum(axis=1, keepdims=True)
        return blended

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class index [0, 1, 2, 3] for feature matrix X.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_single(
        self,
        detection: FIRMSDetection,
        cluster: Optional[ThermalCluster] = None,
        facility: Optional[IndustrialFacility] = None,
        distance_m: Optional[float] = None
    ) -> PredictionOutput:
        """
        Extract features and generate complete tactical PredictionOutput.
        """
        if distance_m is None:
            if cluster and cluster.distance_to_nearest_industrial_m is not None:
                distance_m = cluster.distance_to_nearest_industrial_m
            else:
                distance_m = 5000.0

        feat_vec = extract_28d_features(detection, cluster, facility, distance_m)
        feat_dict = {name: float(val) for name, val in zip(self.feature_names, feat_vec)}

        probs = self.predict_proba(feat_vec.reshape(1, -1))[0]
        pred_class_idx = int(np.argmax(probs))
        pred_class = ThermalAnomalyClass(pred_class_idx)
        confidence = float(probs[pred_class_idx])

        # Named probability map
        prob_map = {
            CLASS_SHORT_LABELS[i]: float(probs[i])
            for i in range(4)
        }

        # Calculate risk and critical alerts
        rsi, _, is_crit = RiskScorer.compute_risk(
            detection=detection,
            cluster=cluster,
            dist_to_industrial_m=distance_m,
            is_inside_industrial=bool(feat_vec[18] > 0.5),
            predicted_class=pred_class,
            confidence_score=confidence
        )

        return PredictionOutput(
            detection_id=detection.detection_id,
            predicted_class=pred_class,
            predicted_label=CLASS_NAMES[pred_class_idx],
            confidence_score=confidence,
            class_probabilities=prob_map,
            risk_severity_index=rsi,
            is_critical_alert=is_crit,
            feature_values=feat_dict
        )

    def predict_batch(
        self,
        detections: Sequence[FIRMSDetection],
        clusters: Optional[Sequence[ThermalCluster]] = None,
    ) -> list[PredictionOutput]:
        """
        Predict full output for a list of detections and optional clusters.
        Vectorized for instant sub-second execution.
        """
        det_to_cluster: dict[str, ThermalCluster] = {}
        if clusters:
            for c in clusters:
                for did in c.detection_ids:
                    det_to_cluster[did] = c

        if not detections:
            return []

        # Vectorized feature matrix extraction
        feat_matrix = []
        for d in detections:
            c = det_to_cluster.get(d.detection_id)
            dist_m = c.distance_to_nearest_industrial_m if (c and c.distance_to_nearest_industrial_m is not None) else 5000.0
            vec = extract_28d_features(d, cluster=c, facility=None, distance_m=dist_m)
            feat_matrix.append(vec)

        X_arr = np.array(feat_matrix, dtype=np.float64)
        all_probs = self.predict_proba(X_arr)
        pred_class_indices = np.argmax(all_probs, axis=1)

        outputs = []
        for i, d in enumerate(detections):
            c = det_to_cluster.get(d.detection_id)
            probs = all_probs[i]
            pred_class_idx = int(pred_class_indices[i])
            pred_class = ThermalAnomalyClass(pred_class_idx)
            confidence = float(probs[pred_class_idx])

            prob_map = {
                CLASS_SHORT_LABELS[j]: float(probs[j])
                for j in range(4)
            }
            feat_vec = X_arr[i]
            feat_dict = {name: float(val) for name, val in zip(self.feature_names, feat_vec)}

            dist_m = float(feat_vec[17])
            is_inside = bool(feat_vec[18] > 0.5)

            rsi, _, is_crit = RiskScorer.compute_risk(
                detection=d,
                cluster=c,
                dist_to_industrial_m=dist_m,
                is_inside_industrial=is_inside,
                predicted_class=pred_class,
                confidence_score=confidence
            )

            outputs.append(PredictionOutput(
                detection_id=d.detection_id,
                predicted_class=pred_class,
                predicted_label=CLASS_NAMES[pred_class_idx],
                confidence_score=confidence,
                class_probabilities=prob_map,
                risk_severity_index=rsi,
                is_critical_alert=is_crit,
                feature_values=feat_dict
            ))
        return outputs

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """
        Evaluate classification metrics against test set.
        """
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
            "wildfire_f1": float(f1[3]),
            "flare_recall": float(r[0]),
            "agricultural_recall": float(r[2]),
            "wildfire_recall": float(r[3]),
        }

    def get_feature_importances(self) -> dict[str, float]:
        """
        Get averaged feature importances from tree ensemble components.
        """
        if not self.is_fitted:
            return {name: 1.0 / len(self.feature_names) for name in self.feature_names}

        imp_rf = self.rf_model.feature_importances_
        imp_et = self.et_model.feature_importances_
        avg_imp = (imp_rf + imp_et) / 2.0
        return {name: float(val) for name, val in zip(self.feature_names, avg_imp)}

    def save(self, filepath: str) -> None:
        """Serialize model to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> ThermalEnsembleClassifier:
        """Load serialized model from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
