"""
Model Training & Cross-Validation Pipeline for NTRO Thermal Detection System.
Trains cost-sensitive multi-class ensemble models and evaluates performance across benchmark corridors.
Authoritative Specifications: ORIGINAL_REQUEST.md § R2, PROJECT.md § 2, TEST_INFRA.md
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

from src.features.feature_extractor import FeatureExtractor, FEATURE_NAMES_28D, extract_28d_features
from src.data_pipeline.schemas import FIRMSDetection, IndustrialFacility
from src.clustering.schemas import ThermalCluster
from src.clustering.dbscan_engine import SpatioTemporalClusteringEngine
from src.data_pipeline.synthetic_generator import SyntheticDataGenerator
from src.classification.ensemble_classifier import ThermalEnsembleClassifier
from src.classification.heuristic_baseline import ThermalAnomalyClass, CLASS_NAMES


class ModelTrainer:
    """
    Orchestrates dataset synthesis, feature matrix compilation,
    cross-validation, and production model fitting.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_extractor = FeatureExtractor()

    def prepare_dataset_from_synthetic(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[FIRMSDetection]]:
        """
        Generate multi-corridor benchmark dataset and compile 28D feature matrix with true class labels.
        """
        gen = SyntheticDataGenerator(seed=self.random_state)
        suite = gen.generate_master_benchmark_suite()

        all_detections: list[FIRMSDetection] = []
        target_labels: list[int] = []

        # Corridor to ground truth class mapping
        corridor_class_map = {
            "jamnagar": ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE.value,
            "jamnagar_refinery": ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE.value,
            "jurong": ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE.value,
            "jurong_island": ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE.value,
            "chemical_disaster": ThermalAnomalyClass.INDUSTRIAL_FIRE_DISASTER.value,
            "punjab_stubble": ThermalAnomalyClass.AGRICULTURAL_STUBBLE_BURNING.value,
            "punjab_agriculture": ThermalAnomalyClass.AGRICULTURAL_STUBBLE_BURNING.value,
            "simlipal_wildfire": ThermalAnomalyClass.WILDFIRE_VEGETATION_BURNING.value,
        }

        for corridor_name, ds in suite.items():
            true_class = corridor_class_map.get(corridor_name, 3)
            for d in ds.detections:
                all_detections.append(d)
                target_labels.append(true_class)

        # Inject dedicated Industrial Fire Disaster examples (Class 1) in industrial facilities
        # to ensure strong training coverage for emergency conditions
        jamnagar_ds = suite.get("jamnagar") or suite.get("jamnagar_refinery")
        if jamnagar_ds and len(jamnagar_ds.facilities) > 0:
            fac = jamnagar_ds.facilities[0]
            c_lat = (fac.bounding_box[1] + fac.bounding_box[3]) / 2.0
            c_lon = (fac.bounding_box[0] + fac.bounding_box[2]) / 2.0
            
            for i in range(40):
                # High FRP, sudden spike, night or day, collocated in refinery
                import datetime
                det_emerg = FIRMSDetection(
                    detection_id=f"SYN_EMERG_{i+1:04d}",
                    latitude=c_lat + (np.random.normal(0, 0.0002)),
                    longitude=c_lon + (np.random.normal(0, 0.0002)),
                    acq_date="2026-03-20",
                    acq_time=f"{18 + (i % 5):02d}30",
                    timestamp=datetime.datetime(2026, 3, 20, 18 + (i % 5), 30),
                    sensor="VIIRS_SNPP",
                    satellite="SNPP",
                    frp=180.0 + (i * 8.0),
                    brightness_temp_t4=365.0 + (i * 2.0),
                    brightness_temp_t11=290.0,
                    bright_delta=75.0 + (i * 2.0),
                    confidence=0.98,
                    daynight="N" if i % 2 == 0 else "D"
                )
                all_detections.append(det_emerg)
                target_labels.append(ThermalAnomalyClass.INDUSTRIAL_FIRE_DISASTER.value)

        # Run clustering on all detections to obtain realistic cluster features
        clustering_engine = SpatioTemporalClusteringEngine()
        clusters = clustering_engine.run_dbscan_clustering(all_detections, eps_m=1000.0, min_samples=3)

        # Extract 28D feature matrix
        X = self.feature_extractor.extract_batch(all_detections, clusters)
        y = np.array(target_labels, dtype=int)

        return X, y, all_detections

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ) -> dict[str, Any]:
        """
        Perform 5-fold Stratified Cross-Validation on the dataset.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            clf = ThermalEnsembleClassifier(random_state=self.random_state + fold)
            clf.fit(X_train, y_train)
            metrics = clf.evaluate(X_val, y_val)
            fold_metrics.append(metrics)

        # Aggregate across folds
        avg_metrics = {}
        for key in fold_metrics[0].keys():
            vals = [m[key] for m in fold_metrics]
            avg_metrics[f"mean_{key}"] = float(np.mean(vals))
            avg_metrics[f"std_{key}"] = float(np.std(vals))

        return {
            "n_splits": n_splits,
            "fold_results": fold_metrics,
            "summary": avg_metrics
        }

    def train_and_save_model(
        self,
        output_path: str = "data/models/thermal_classifier.pkl"
    ) -> tuple[ThermalEnsembleClassifier, dict[str, float]]:
        """
        Train production ensemble on full benchmark suite and save model.
        """
        X, y, _ = self.prepare_dataset_from_synthetic()
        
        # Split train / test for final verification
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        train_idx, test_idx = next(skf.split(X, y))
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = ThermalEnsembleClassifier(random_state=self.random_state)
        model.fit(X_train, y_train)
        eval_results = model.evaluate(X_test, y_test)

        # Fit on full dataset for production deployment
        full_model = ThermalEnsembleClassifier(random_state=self.random_state)
        full_model.fit(X, y)
        full_model.save(output_path)

        return full_model, eval_results
