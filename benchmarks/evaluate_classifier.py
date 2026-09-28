"""
ML Classifier Benchmark & Baseline Comparison Suite.
Evaluates Multi-Class Thermal Ensemble against Heuristic Baseline across all Acceptance Criteria.
Authoritative Specifications: ORIGINAL_REQUEST.md § Acceptance Criteria, PROJECT.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.classification.ensemble_classifier import ThermalEnsembleClassifier
from src.classification.heuristic_baseline import HeuristicBaselineClassifier
from src.classification.model_trainer import ModelTrainer
from src.explainability.shap_engine import SHAPExplainerEngine


def run_classifier_benchmark() -> dict[str, Any]:
    """Execute master classifier benchmark and compare against baseline heuristics."""
    print("=" * 75)
    print("NTRO THERMAL CLASSIFIER & XAI BENCHMARK SUITE (SIH26162)")
    print("=" * 75)

    trainer = ModelTrainer(random_state=42)
    print("[1/4] Synthesizing multi-corridor benchmark dataset and extracting 28D features...")
    X, y, detections = trainer.prepare_dataset_from_synthetic()
    print(f"      Total dataset size: {len(X)} instances across 4 classes.")
    print(f"      Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 5-fold cross-validation split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(skf.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # 1. Evaluate Heuristic Baseline
    print("\n[2/4] Evaluating Deterministic Heuristic Baseline Classifier...")
    baseline = HeuristicBaselineClassifier()
    t0 = time.perf_counter()
    baseline_metrics = baseline.evaluate(X_test, y_test)
    t_baseline = (time.perf_counter() - t0) * 1000.0 / len(X_test)

    # 2. Train and Evaluate Thermal Ensemble Classifier
    print("\n[3/4] Training & Evaluating Cost-Sensitive Thermal Ensemble Classifier...")
    ensemble = ThermalEnsembleClassifier(random_state=42)
    ensemble.fit(X_train, y_train)
    t0 = time.perf_counter()
    ensemble_metrics = ensemble.evaluate(X_test, y_test)
    t_ensemble = (time.perf_counter() - t0) * 1000.0 / len(X_test)

    # 3. Generate SHAP Summary
    print("\n[4/4] Generating SHAP TreeExplainer Attribution and Global Summary...")
    explainer = SHAPExplainerEngine(classifier=ensemble, background_data=X_train[:30])
    summary_plot_path = "data/reports/shap_global_summary.png"
    explainer.generate_global_summary_plot(X_test, output_path=summary_plot_path)
    print(f"      Global SHAP summary chart generated at: {summary_plot_path}")

    # Display benchmark comparison
    print("\n" + "=" * 75)
    print("BENCHMARK COMPARISON TABLE: HEURISTIC BASELINE vs. THERMAL ENSEMBLE")
    print("=" * 75)
    print(f"{'Metric':<32} | {'Heuristic Baseline':<18} | {'Thermal Ensemble':<18} | {'Target':<10}")
    print("-" * 75)
    print(f"{'Macro F1-Score':<32} | {baseline_metrics['macro_f1']:.4f}{'':<12} | {ensemble_metrics['macro_f1']:.4f}{'':<12} | >= 0.8500")
    print(f"{'Industrial Emergency Recall (C1)':<32} | {baseline_metrics['emergency_recall']:.4f}{'':<12} | {ensemble_metrics['emergency_recall']:.4f}{'':<12} | >= 0.9000")
    print(f"{'Overall Accuracy':<32} | {baseline_metrics['accuracy']:.4f}{'':<12} | {ensemble_metrics['accuracy']:.4f}{'':<12} | -")
    print(f"{'Controlled Flare F1':<32} | {baseline_metrics['flare_f1']:.4f}{'':<12} | {ensemble_metrics['flare_f1']:.4f}{'':<12} | -")
    print(f"{'Industrial Emergency F1':<32} | {baseline_metrics['emergency_f1']:.4f}{'':<12} | {ensemble_metrics['emergency_f1']:.4f}{'':<12} | -")
    print(f"{'Agricultural Stubble F1':<32} | {baseline_metrics['agricultural_f1']:.4f}{'':<12} | {ensemble_metrics['agricultural_f1']:.4f}{'':<12} | -")
    print(f"{'Wildfire Burning F1':<32} | {baseline_metrics['wildfire_f1']:.4f}{'':<12} | {ensemble_metrics['wildfire_f1']:.4f}{'':<12} | -")
    print(f"{'Inference Latency (ms/sample)':<32} | {t_baseline:.3f} ms{'':<10} | {t_ensemble:.3f} ms{'':<10} | < 10.0 ms")
    print("=" * 75)

    f1_pass = ensemble_metrics["macro_f1"] >= 0.85
    recall_pass = ensemble_metrics["emergency_recall"] >= 0.90
    all_pass = f1_pass and recall_pass

    print(f"VERDICT: {'PASSED [AC-ML-1 & AC-ML-2 SATISFIED]' if all_pass else 'FAILED'}")
    print("=" * 75)

    # Save trained model to data/models/
    ensemble.save("data/models/thermal_classifier.pkl")

    return {
        "baseline_metrics": baseline_metrics,
        "ensemble_metrics": ensemble_metrics,
        "f1_pass": f1_pass,
        "recall_pass": recall_pass,
    }


if __name__ == "__main__":
    res = run_classifier_benchmark()
    if not (res["f1_pass"] and res["recall_pass"]):
        sys.exit(1)
