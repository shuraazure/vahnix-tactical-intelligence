"""
Milestone 3 Test Suite: Multi-Class ML Classification & SHAP XAI Engine.
Validates 28D Feature Extraction, Risk Scorer, Heuristic Baseline, Cost-Sensitive Ensemble,
SHAP Explainer Engine, and Natural Language Tactical XAI.
Authoritative Specifications: ORIGINAL_REQUEST.md § R2, PROJECT.md § 2, TEST_INFRA.md
"""

from __future__ import annotations

import datetime
import math
import os
import tempfile
import numpy as np
import pytest

from src.data_pipeline.schemas import FIRMSDetection, IndustrialFacility
from src.clustering.schemas import ThermalCluster, ClusterType
from src.features.feature_extractor import (
    FeatureExtractor,
    FEATURE_NAMES_28D,
    extract_28d_features,
)
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
from src.explainability.shap_engine import (
    SHAPExplanation,
    SHAPExplainerEngine,
)
from src.explainability.natural_language_xai import NaturalLanguageXAI


# =========================================================================
# 1. 28D FEATURE EXTRACTION TESTS
# =========================================================================

def test_feature_extractor_28d_dimension_and_invariants():
    """Verify feature extractor produces exact 28D vector with valid float types."""
    now = datetime.datetime.utcnow()
    det = FIRMSDetection(
        detection_id="DET_TEST_001",
        latitude=22.3582,
        longitude=69.8681,
        acq_date="2026-03-15",
        acq_time="1200",
        timestamp=now,
        sensor="VIIRS_SNPP",
        satellite="SNPP",
        frp=45.0,
        brightness_temp_t4=340.0,
        brightness_temp_t11=295.0,
        bright_delta=45.0,
        confidence=0.92,
        daynight="D"
    )

    vec = extract_28d_features(det, distance_m=120.0)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (28,)
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()

    # Check key indices
    assert math.isclose(vec[0], 45.0)   # frp
    assert math.isclose(vec[1], 340.0)  # brightness
    assert math.isclose(vec[3], 45.0)   # delta_t
    assert math.isclose(vec[17], 120.0) # dist_to_industrial_m
    assert math.isclose(vec[27], 1.0)   # daynight_binary ('D' -> 1.0)


def test_feature_extractor_batch_and_dataframe():
    """Verify batch extraction and dataframe creation with detection_id column."""
    now = datetime.datetime.utcnow()
    dets = [
        FIRMSDetection(
            detection_id=f"DET_{i}",
            latitude=22.35 + (i * 0.01),
            longitude=69.86 + (i * 0.01),
            acq_date="2026-03-15",
            acq_time="1200",
            timestamp=now,
            sensor="MODIS",
            satellite="Terra",
            frp=20.0 + (i * 5.0),
            brightness_temp_t4=330.0,
            brightness_temp_t11=295.0,
            bright_delta=35.0,
            confidence=0.80,
            daynight="D" if i % 2 == 0 else "N"
        )
        for i in range(5)
    ]

    extractor = FeatureExtractor()
    df = extractor.extract_features_dataframe(dets)
    assert len(df) == 5
    assert "detection_id" in df.columns
    assert len(df.columns) == 29  # 28 features + detection_id


# =========================================================================
# 2. RISK SCORER & ALERT LEVEL TESTS
# =========================================================================

def test_risk_severity_index_formula():
    """Verify composite Risk Severity Index (RSI) calculation and alert boundaries."""
    # Routine flare: moderate FRP, inside industrial, low surge
    rsi_routine = calculate_risk_severity_index(
        frp=35.0,
        bright_delta=30.0,
        dist_to_industrial_m=0.0,
        is_inside_industrial=True,
        frp_zscore=0.2
    )
    assert 20.0 <= rsi_routine <= 45.0

    # Major industrial emergency: 350 MW FRP, 85K Delta-T, inside refinery, surge +4.5
    rsi_emergency = calculate_risk_severity_index(
        frp=350.0,
        bright_delta=85.0,
        dist_to_industrial_m=0.0,
        is_inside_industrial=True,
        frp_zscore=4.5
    )
    assert rsi_emergency >= 80.0


def test_risk_scorer_critical_alert_triggers():
    """Verify Critical Alert logic for emergency conditions."""
    now = datetime.datetime.utcnow()
    det = FIRMSDetection(
        detection_id="EMERG_001",
        latitude=22.3582,
        longitude=69.8681,
        acq_date="2026-03-15",
        acq_time="2200",
        timestamp=now,
        sensor="VIIRS_SNPP",
        satellite="SNPP",
        frp=250.0,
        brightness_temp_t4=370.0,
        brightness_temp_t11=290.0,
        bright_delta=80.0,
        confidence=0.99,
        daynight="N"
    )

    rsi, level, is_crit = RiskScorer.compute_risk(
        detection=det,
        dist_to_industrial_m=50.0,
        is_inside_industrial=True,
        predicted_class=ThermalAnomalyClass.INDUSTRIAL_FIRE_DISASTER,
        confidence_score=0.98
    )

    assert rsi >= 75.0
    assert level in (AlertLevel.ORANGE, AlertLevel.RED)
    assert is_crit is True


# =========================================================================
# 3. HEURISTIC BASELINE TESTS
# =========================================================================

def test_heuristic_baseline_all_classes():
    """Verify heuristic baseline classifies all 4 target scenarios correctly."""
    # Class 0: Controlled Flare (Inside refinery, high recurrence, normal FRP)
    flare_feat = np.zeros(28)
    flare_feat[0] = 30.0    # frp
    flare_feat[17] = 50.0   # dist_to_industrial_m
    flare_feat[18] = 1.0    # is_inside
    flare_feat[10] = 5.0    # recurrence_freq
    assert heuristic_baseline_classify(flare_feat) == ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE

    # Class 1: Industrial Fire Disaster (Inside refinery, extreme FRP)
    emerg_feat = np.zeros(28)
    emerg_feat[0] = 200.0   # frp
    emerg_feat[17] = 50.0   # dist_to_industrial_m
    emerg_feat[18] = 1.0    # is_inside
    emerg_feat[4] = 4.0     # frp_zscore
    assert heuristic_baseline_classify(emerg_feat) == ThermalAnomalyClass.INDUSTRIAL_FIRE_DISASTER

    # Class 2: Agricultural Stubble (Far from industry, daytime, short span)
    agri_feat = np.zeros(28)
    agri_feat[0] = 25.0
    agri_feat[17] = 8000.0  # dist_to_industrial_m
    agri_feat[11] = 4.0     # day_night_ratio
    agri_feat[9] = 2.0      # temporal_span
    assert heuristic_baseline_classify(agri_feat) == ThermalAnomalyClass.AGRICULTURAL_STUBBLE_BURNING

    # Class 3: Wildfire (Far from industry, large radius)
    wild_feat = np.zeros(28)
    wild_feat[0] = 60.0
    wild_feat[17] = 15000.0 # dist_to_industrial_m
    wild_feat[14] = 1500.0  # cluster_radius_m
    assert heuristic_baseline_classify(wild_feat) == ThermalAnomalyClass.WILDFIRE_VEGETATION_BURNING


# =========================================================================
# 4. ENSEMBLE CLASSIFIER & CROSS-VALIDATION TESTS
# =========================================================================

def test_ensemble_classifier_training_and_serialization():
    """Verify ModelTrainer trains ensemble with Macro F1 >= 0.85 and Class 1 Recall >= 0.90."""
    trainer = ModelTrainer(random_state=42)
    X, y, dets = trainer.prepare_dataset_from_synthetic()
    assert len(X) >= 100

    # Cross-validation
    cv_res = trainer.cross_validate(X, y, n_splits=3)
    assert cv_res["summary"]["mean_macro_f1"] >= 0.80

    # Train model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.pkl")
        clf, eval_metrics = trainer.train_and_save_model(output_path=model_path)

        assert eval_metrics["macro_f1"] >= 0.85
        assert eval_metrics["emergency_recall"] >= 0.90

        # Test model loading
        loaded_clf = ThermalEnsembleClassifier.load(model_path)
        assert loaded_clf.is_fitted is True

        # Test single prediction
        pred_out = loaded_clf.predict_single(dets[0])
        assert isinstance(pred_out, PredictionOutput)
        assert pred_out.confidence_score > 0.50
        assert sum(pred_out.class_probabilities.values()) == pytest.approx(1.0, abs=1e-3)


# =========================================================================
# 5. SHAP EXPLAINER & NATURAL LANGUAGE XAI TESTS
# =========================================================================

def test_shap_explainer_local_attribution_and_summary():
    """Verify SHAP explanation generation, feature attribution, and natural language summary."""
    trainer = ModelTrainer(random_state=42)
    X, y, dets = trainer.prepare_dataset_from_synthetic()
    clf = ThermalEnsembleClassifier(random_state=42)
    clf.fit(X, y)

    explainer = SHAPExplainerEngine(classifier=clf, background_data=X[:20])
    pred_out = clf.predict_single(dets[0])

    exp = explainer.explain_instance(X[0], pred_out, generate_plot=True)
    assert isinstance(exp, SHAPExplanation)
    assert exp.detection_id == dets[0].detection_id
    assert len(exp.shap_values) == 28
    assert len(exp.top_positive_features) >= 1
    assert exp.waterfall_plot_base64 is not None and len(exp.waterfall_plot_base64) > 100
    assert len(exp.natural_language_summary) > 50

    # Verify global summary plot generation
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = os.path.join(tmpdir, "shap_summary.png")
        explainer.generate_global_summary_plot(X[:30], output_path=plot_path)
        assert os.path.exists(plot_path)
        assert os.path.getsize(plot_path) > 1000
