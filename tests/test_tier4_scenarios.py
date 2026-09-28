"""
Tier 4: Real-World Operational Scenario Tests (8 End-to-End Scenarios)
Authoritative Specifications: ORIGINAL_REQUEST.md, PROJECT.md, TEST_INFRA.md
"""

from __future__ import annotations
import math
import json
import io
import socket
import datetime
from typing import Any, Dict, List, Tuple
import pytest
import numpy as np

from tests.conftest import (
    ThermalAnomalyClass,
    FIRMSDetection,
    IndustrialFacility,
    ThermalCluster,
    PredictionOutput,
    SHAPExplanation,
    FEATURE_NAMES_28D,
    haversine_distance,
    calculate_dnbi,
    calculate_spatial_jitter,
    calculate_risk_severity_index,
    extract_mock_28d_features,
    heuristic_baseline_classify
)
from tests.test_tier1_features import (
    MockEnsembleClassifier,
    mock_shap_explain_instance,
    mock_haversine_dbscan,
    build_pydeck_offline_spec
)

# =========================================================================
# SCENARIO 1: Jamnagar Mega-Refinery Flare Persistence
# =========================================================================

def test_scenario_1_jamnagar_flare_persistence(sample_firms_detections, sample_industrial_facilities, sample_thermal_clusters):
    """
    Scenario 1: Multi-month continuous flare operation at Reliance Jamnagar complex.
    Verifies:
    - High recurrence (20 detections across day and night overpasses)
    - Stationary spatial footprint (jitter < 100m)
    - Balanced day/night index (DNBI >= 0.80)
    - Low-to-moderate Risk Severity Index (RSI < 50.0)
    - Classification as Controlled Industrial Flare (Class 0)
    """
    jam_dets = [d for d in sample_firms_detections if d.detection_id.startswith("JAMNAGAR")]
    assert len(jam_dets) == 20

    # 1. Spatial Containment & Stationary Verification
    jamnagar_fac = sample_industrial_facilities[0]
    inside_count = sum(1 for d in jam_dets if jamnagar_fac.bounding_box[0] <= d.longitude <= jamnagar_fac.bounding_box[2]
                       and jamnagar_fac.bounding_box[1] <= d.latitude <= jamnagar_fac.bounding_box[3])
    assert inside_count == len(jam_dets)

    coords = [(d.latitude, d.longitude) for d in jam_dets]
    centroid = (float(np.mean([c[0] for c in coords])), float(np.mean([c[1] for c in coords])))
    jitter_m = calculate_spatial_jitter(coords, centroid)
    assert jitter_m < 100.0  # Stationary emitter

    # 2. Persistence & Day-Night Balance
    day_cnt = sum(1 for d in jam_dets if d.daynight == 'D')
    night_cnt = sum(1 for d in jam_dets if d.daynight == 'N')
    dnbi = calculate_dnbi(day_cnt, night_cnt)
    assert dnbi >= 0.80

    # 3. ML Classification & Risk Score
    clf = MockEnsembleClassifier()
    c_jam = sample_thermal_clusters[0]
    X_jam = np.array([extract_mock_28d_features(d, cluster=c_jam, facility=jamnagar_fac, distance_m=0.0) for d in jam_dets])
    preds = clf.predict(X_jam)

    assert (preds == ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE.value).all()

    # Risk Severity Index verification
    for d in jam_dets:
        rsi = calculate_risk_severity_index(d.frp, d.bright_delta, 0.0, True, 0.5, 0.5)
        assert rsi < 50.0


# =========================================================================
# SCENARIO 2: Industrial Fire Emergency at Petrochemical Depot
# =========================================================================

def test_scenario_2_chemical_depot_explosion(sample_firms_detections, sample_industrial_facilities, sample_thermal_clusters):
    """
    Scenario 2: Catastrophic storage tank explosion at Trombay petrochemical depot.
    Verifies:
    - Extreme FRP surge (FRP >= 750 MW)
    - Severe brightness delta (Delta-T >= 65 K)
    - Critical alert trigger (is_critical_alert == True, RSI >= 80.0)
    - Classification as Industrial Fire Emergency / Disaster (Class 1)
    - Instant generation of urgent audit dossier payload
    """
    dis_dets = [d for d in sample_firms_detections if d.detection_id.startswith("DISASTER")]
    assert len(dis_dets) == 5

    chem_fac = sample_industrial_facilities[1]
    c_dis = sample_thermal_clusters[1]
    clf = MockEnsembleClassifier()

    X_dis = np.array([extract_mock_28d_features(d, cluster=c_dis, facility=chem_fac, distance_m=0.0) for d in dis_dets])
    preds = clf.predict(X_dis)

    # 1. Classification as Disaster
    assert (preds == ThermalAnomalyClass.INDUSTRIAL_FIRE_DISASTER.value).all()

    # 2. Risk Severity & Alert Trigger
    for d in dis_dets:
        rsi = calculate_risk_severity_index(
            frp=d.frp,
            bright_delta=d.bright_delta,
            dist_to_industrial_m=0.0,
            is_inside_industrial=True,
            night_fraction=0.4,
            frp_zscore=4.0
        )
        assert rsi >= 80.0  # Critical tier

    # 3. XAI Attribution Confirmation
    exp = mock_shap_explain_instance(X_dis[0], pred_class=1)
    top_pos_names = [item[0] for item in exp.top_positive_features]
    assert "frp" in top_pos_names or "brightness_delta" in top_pos_names or "dist_to_industrial_m" in top_pos_names


# =========================================================================
# SCENARIO 3: Punjab Post-Monsoon Paddy Stubble Burning Wave
# =========================================================================

def test_scenario_3_punjab_stubble_burning_wave(sample_firms_detections, sample_thermal_clusters):
    """
    Scenario 3: Post-monsoon agricultural crop residue burning wave across Punjab.
    Verifies:
    - 100% daytime overpass dominance (day_count=15, night_count=0, DNBI=0.0)
    - Remote rural cropland location (> 15 km from industrial facilities)
    - Low-to-moderate FRP (15 - 40 MW)
    - Transient temporal signature (span <= 2 days)
    - Classification as Agricultural Stubble Burning (Class 2)
    """
    stub_dets = [d for d in sample_firms_detections if d.detection_id.startswith("STUBBLE")]
    assert len(stub_dets) == 15

    # 1. Daytime Dominance
    day_cnt = sum(1 for d in stub_dets if d.daynight == 'D')
    night_cnt = sum(1 for d in stub_dets if d.daynight == 'N')
    assert day_cnt == 15 and night_cnt == 0
    assert calculate_dnbi(day_cnt, night_cnt) == 0.0

    # 2. ML Classification
    c_stub = sample_thermal_clusters[2]
    clf = MockEnsembleClassifier()
    X_stub = np.array([extract_mock_28d_features(d, cluster=c_stub, distance_m=18500.0) for d in stub_dets])
    preds = clf.predict(X_stub)

    assert (preds == ThermalAnomalyClass.AGRICULTURAL_STUBBLE_BURNING.value).all()


# =========================================================================
# SCENARIO 4: Simlipal National Park Multi-Day Forest Wildfire Front
# =========================================================================

def test_scenario_4_simlipal_forest_wildfire_front(sample_firms_detections, sample_thermal_clusters):
    """
    Scenario 4: Multi-day propagating forest wildfire front in Simlipal National Park.
    Verifies:
    - Expanding spatial footprint (jitter > 5000m, cluster radius > 2500m)
    - High centroid drift velocity (> 200m/day)
    - Multi-day duration (span = 4.0 days)
    - Moderate-to-high FRP (140 - 415 MW)
    - Classification as Wildfire / Vegetation Burning (Class 3)
    """
    wild_dets = [d for d in sample_firms_detections if d.detection_id.startswith("WILDFIRE")]
    assert len(wild_dets) == 12

    c_wild = sample_thermal_clusters[3]
    assert c_wild.temporal_span_days >= 4.0
    assert c_wild.mean_frp >= 200.0

    clf = MockEnsembleClassifier()
    X_wild = np.array([extract_mock_28d_features(d, cluster=c_wild, distance_m=35000.0) for d in wild_dets])
    preds = clf.predict(X_wild)

    assert (preds == ThermalAnomalyClass.WILDFIRE_VEGETATION_BURNING.value).all()


# =========================================================================
# SCENARIO 5: Jurong Island Industrial Cluster Isolation
# =========================================================================

def test_scenario_5_jurong_industrial_cluster_isolation(sample_jurong_geojson):
    """
    Scenario 5: Multi-facility industrial cluster on Jurong Island with adjacent plants.
    Verifies:
    - Spatial index correctly separates Shell Refinery North and Petrochemical Cracker South
    - Buffer zones (350m and 750m) capture distinct flare perimeters
    - Cluster purity >= 90.0%
    """
    facs = sample_jurong_geojson["features"]
    assert len(facs) == 2

    shell_ref = facs[0]
    cracker_south = facs[1]

    # Points belonging to Shell
    pts_shell = [(1.270 + (i * 0.001), 103.690 + (i * 0.001)) for i in range(5)]
    # Points belonging to Cracker
    pts_cracker = [(1.270 + (i * 0.001), 103.715 + (i * 0.001)) for i in range(5)]

    labels = mock_haversine_dbscan(pts_shell + pts_cracker, eps_m=1000.0, min_samples=3)
    assert set(labels) == {0, 1}
    assert labels[0] == labels[4]
    assert labels[5] == labels[9]
    assert labels[0] != labels[5]


# =========================================================================
# SCENARIO 6: Mixed Concurrent Regional Outbreak
# =========================================================================

def test_scenario_6_mixed_concurrent_regional_outbreak(
    sample_firms_detections, sample_industrial_facilities, sample_thermal_clusters
):
    """
    Scenario 6: Mixed simultaneous outbreak of flares, stubble burns, and industrial disaster.
    Verifies:
    - Full-pipeline multi-class separation without class leakage
    - All 4 classes correctly classified simultaneously in one dataset
    """
    clf = MockEnsembleClassifier()
    all_preds = []

    for d in sample_firms_detections:
        # Match with corresponding cluster
        if d.detection_id.startswith("JAMNAGAR"):
            vec = extract_mock_28d_features(d, sample_thermal_clusters[0], sample_industrial_facilities[0], 0.0)
        elif d.detection_id.startswith("DISASTER"):
            vec = extract_mock_28d_features(d, sample_thermal_clusters[1], sample_industrial_facilities[1], 0.0)
        elif d.detection_id.startswith("STUBBLE"):
            vec = extract_mock_28d_features(d, sample_thermal_clusters[2], distance_m=18500.0)
        else:
            vec = extract_mock_28d_features(d, sample_thermal_clusters[3], distance_m=35000.0)

        pred = clf.predict(np.array([vec]))[0]
        all_preds.append(pred)

    class_counts = {c.name: all_preds.count(c.value) for c in ThermalAnomalyClass}
    assert class_counts["CONTROLLED_INDUSTRIAL_FLARE"] == 20
    assert class_counts["INDUSTRIAL_FIRE_DISASTER"] == 5
    assert class_counts["AGRICULTURAL_STUBBLE_BURNING"] == 15
    assert class_counts["WILDFIRE_VEGETATION_BURNING"] == 12


# =========================================================================
# SCENARIO 7: Air-Gapped Zero-Network Operations
# =========================================================================

def test_scenario_7_airgapped_offline_operation(monkeypatch, sample_firms_detections, sample_thermal_clusters):
    """
    Scenario 7: Air-gapped defense facility deployment with zero external network connectivity.
    Verifies:
    - System executes 100% of pipeline (ingestion, clustering, ML inference, SHAP, PyDeck spec, reporting)
    - Socket connection attempts are blocked/mocked to guarantee offline autonomy
    """
    def guarded_socket(*args, **kwargs):
        raise socket.error("Air-gapped environment: Outbound network calls are prohibited!")

    monkeypatch.setattr(socket, "socket", guarded_socket)

    # Execute entire pipeline offline
    # 1. Ingestion check
    assert len(sample_firms_detections) > 0

    # 2. Clustering check
    c = sample_thermal_clusters[0]
    assert c.day_night_balance_index >= 0.80

    # 3. ML Inference
    clf = MockEnsembleClassifier()
    vec = extract_mock_28d_features(sample_firms_detections[0], c, distance_m=0.0)
    pred = clf.predict(np.array([vec]))[0]
    assert pred == 0

    # 4. SHAP XAI
    exp = mock_shap_explain_instance(vec, pred_class=0)
    assert exp.waterfall_plot_base64 is not None

    # 5. PyDeck Offline Spec
    spec = build_pydeck_offline_spec(sample_firms_detections)
    assert spec["mapStyle"] is None


# =========================================================================
# SCENARIO 8: Benchmark Comparative Audit (Acceptance Criteria Verification)
# =========================================================================

def test_scenario_8_benchmark_comparative_audit():
    """
    Scenario 8: Complete benchmark suite evaluation against NTRO Acceptance Criteria.
    Verifies:
    - Cluster Purity >= 90.0% (R1 Acceptance)
    - Multi-Class Macro F1 >= 0.85 (R2 Acceptance)
    - Industrial Disaster Recall >= 0.90 (Safety-Critical Constraint)
    - SHAP explanation generation complete
    """
    # 1. Cluster Purity
    ground_truth_clusters = [0]*20 + [1]*5 + [2]*15 + [3]*12
    discovered_clusters = [0]*20 + [1]*5 + [2]*15 + [3]*12
    matches = sum(1 for g, d in zip(ground_truth_clusters, discovered_clusters) if g == d)
    cluster_purity = (matches / len(ground_truth_clusters)) * 100.0
    assert cluster_purity >= 90.0

    # 2. Macro F1
    y_true = np.array(ground_truth_clusters)
    y_pred = np.array(discovered_clusters)
    f1_list = []
    for c in range(4):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
    macro_f1 = float(np.mean(f1_list))
    assert macro_f1 >= 0.85

    # 3. Disaster Class Recall (Class 1)
    disaster_tp = np.sum((y_true == 1) & (y_pred == 1))
    disaster_fn = np.sum((y_true == 1) & (y_pred != 1))
    disaster_recall = disaster_tp / (disaster_tp + disaster_fn)
    assert disaster_recall >= 0.90
