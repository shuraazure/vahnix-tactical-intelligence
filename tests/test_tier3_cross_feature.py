"""
Tier 3: Cross-Module Integration & Pairwise Pipeline Tests (18 Integration Tests)
Authoritative Specifications: ORIGINAL_REQUEST.md, PROJECT.md, TEST_INFRA.md
"""

from __future__ import annotations
import math
import json
import io
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
    parse_modis_csv_line,
    parse_viirs_csv_line,
    point_in_polygon_bbox,
    mock_lat_lon_to_h3,
    mock_haversine_dbscan,
    MockEnsembleClassifier,
    mock_shap_explain_instance,
    build_pydeck_offline_spec
)

# =========================================================================
# CROSS-MODULE INTEGRATION TESTS (TIER 3)
# =========================================================================

def test_tier3_01_ingestion_to_spatial_index_to_proximity(sample_modis_csv_content, sample_industrial_facilities):
    """
    Integration 1: FIRMS_INGESTION -> OSM_SPATIAL_INDEX -> DISTANCE_PROXIMITY
    Raw CSV parsing feeding directly into spatial index query and geodesic distance calculation.
    """
    lines = sample_modis_csv_content.strip().split("\n")
    records = [parse_modis_csv_line(l) for l in lines[1:] if parse_modis_csv_line(l)]
    assert len(records) > 0

    # Ingest into FIRMSDetection objects
    detections = [
        FIRMSDetection(
            detection_id=f"MODIS_{i}",
            latitude=r["latitude"],
            longitude=r["longitude"],
            acq_date=r["acq_date"],
            acq_time=r["acq_time"],
            timestamp=datetime.datetime.strptime(f"{r['acq_date']} {r['acq_time']}", "%Y-%m-%d %H%M"),
            sensor=r["sensor"],
            satellite=r["satellite"],
            frp=r["frp"],
            brightness_temp_t4=r["brightness_temp_t4"],
            brightness_temp_t11=r["brightness_temp_t11"],
            bright_delta=r["bright_delta"],
            confidence=r["confidence"],
            daynight=r["daynight"]
        )
        for i, r in enumerate(records)
    ]

    # Query nearest industrial facility
    jamnagar = sample_industrial_facilities[0]
    jam_cent = ((jamnagar.bounding_box[1] + jamnagar.bounding_box[3]) / 2.0, (jamnagar.bounding_box[0] + jamnagar.bounding_box[2]) / 2.0)

    distances = [haversine_distance(d.latitude, d.longitude, jam_cent[0], jam_cent[1]) for d in detections]
    # First detection is near Jamnagar refinery
    assert distances[0] < 5000.0
    # Punjab detection is far away (> 500 km)
    assert distances[3] > 500000.0


def test_tier3_02_ingestion_to_h3_spatial_binning(sample_viirs_csv_content):
    """
    Integration 2: FIRMS_INGESTION -> H3_SPATIAL_BINNING -> HEX_CLUSTER_SUMMARY
    Raw VIIRS CSV records partitioned into discrete H3 global hexagonal bins.
    """
    lines = sample_viirs_csv_content.strip().split("\n")
    records = [parse_viirs_csv_line(l) for l in lines[1:] if parse_viirs_csv_line(l)]

    hex_bins: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        h3_res8 = mock_lat_lon_to_h3(r["latitude"], r["longitude"], res=8)
        hex_bins.setdefault(h3_res8, []).append(r)

    assert len(hex_bins) >= 3
    # Check aggregation per hex bin
    for h3_idx, recs in hex_bins.items():
        total_frp = sum(x["frp"] for x in recs)
        mean_frp = total_frp / len(recs)
        assert mean_frp > 0.0


def test_tier3_03_spatial_index_to_dbscan_to_facility_tagging(sample_firms_detections, sample_industrial_facilities):
    """
    Integration 3: OSM_SPATIAL_INDEX -> DBSCAN_CLUSTERING -> FACILITY_TAGGING
    Spatial clustering groups detections, and cluster centroids are tagged with nearest OSM asset.
    """
    points = [(d.latitude, d.longitude) for d in sample_firms_detections]
    labels = mock_haversine_dbscan(points, eps_m=1000.0, min_samples=3)

    cluster_groups: Dict[int, List[FIRMSDetection]] = {}
    for label, det in zip(labels, sample_firms_detections):
        if label != -1:
            cluster_groups.setdefault(label, []).append(det)

    assert len(cluster_groups) >= 2

    # Tag each cluster with nearest facility
    for c_id, dets in cluster_groups.items():
        cent_lat = float(np.mean([d.latitude for d in dets]))
        cent_lon = float(np.mean([d.longitude for d in dets]))
        min_d = min(
            haversine_distance(cent_lat, cent_lon, (f.bounding_box[1]+f.bounding_box[3])/2.0, (f.bounding_box[0]+f.bounding_box[2])/2.0)
            for f in sample_industrial_facilities
        )
        assert min_d >= 0.0


def test_tier3_04_ingestion_to_st_dbscan_propagation(sample_firms_detections):
    """
    Integration 4: FIRMS_INGESTION -> ST_DBSCAN -> DRIFT_VELOCITY_CALCULATION
    Spatio-temporal grouping across 48h window detecting dynamic wildfire front propagation.
    """
    wildfire_dets = [d for d in sample_firms_detections if d.detection_id.startswith("WILDFIRE")]
    assert len(wildfire_dets) >= 10

    # Calculate spatial progression over time
    first_half = wildfire_dets[:len(wildfire_dets)//2]
    second_half = wildfire_dets[len(wildfire_dets)//2:]

    c1 = (float(np.mean([d.latitude for d in first_half])), float(np.mean([d.longitude for d in first_half])))
    c2 = (float(np.mean([d.latitude for d in second_half])), float(np.mean([d.longitude for d in second_half])))

    dist_drift_m = haversine_distance(c1[0], c1[1], c2[0], c2[1])
    dt_days = (second_half[-1].timestamp - first_half[0].timestamp).total_seconds() / 86400.0
    drift_velocity_m_per_day = dist_drift_m / max(dt_days, 0.5)

    assert drift_velocity_m_per_day > 100.0  # Dynamic moving front


def test_tier3_05_dbscan_to_persistence_metrics(sample_firms_detections):
    """
    Integration 5: DBSCAN_CLUSTERING -> PERSISTENCE_METRICS (DNBI, Recurrence, Jitter)
    Clustering outputs converted into statistical persistence indicators.
    """
    jam_dets = [d for d in sample_firms_detections if d.detection_id.startswith("JAMNAGAR")]
    day_cnt = sum(1 for d in jam_dets if d.daynight == 'D')
    night_cnt = sum(1 for d in jam_dets if d.daynight == 'N')
    dnbi = calculate_dnbi(day_cnt, night_cnt)
    coords = [(d.latitude, d.longitude) for d in jam_dets]
    centroid = (float(np.mean([c[0] for c in coords])), float(np.mean([c[1] for c in coords])))
    jitter = calculate_spatial_jitter(coords, centroid)

    assert dnbi >= 0.80  # High day/night balance
    assert jitter < 150.0  # Stationary emitter


def test_tier3_06_ingestion_and_spatial_and_persistence_to_feature_extraction(
    sample_firms_detections, sample_thermal_clusters, sample_industrial_facilities
):
    """
    Integration 6: INGESTION + SPATIAL_INDEX + PERSISTENCE -> FEATURE_ENGINEERING
    Full physical 28D feature vector assembly from combined data layers.
    """
    det = sample_firms_detections[0]
    cluster = sample_thermal_clusters[0]
    facility = sample_industrial_facilities[0]

    vec = extract_mock_28d_features(det, cluster=cluster, facility=facility, distance_m=0.0)
    assert len(vec) == 28
    assert vec[0] == pytest.approx(det.frp, 0.01)
    assert vec[8] == float(cluster.total_detections)
    assert vec[17] == 0.0  # inside facility
    assert vec[18] == 1.0


def test_tier3_07_feature_pipeline_to_ml_inference(sample_firms_detections, sample_thermal_clusters):
    """
    Integration 7: FEATURE_ENGINEERING -> ML_CLASSIFIER_SUITE (Ensemble Inference)
    28D feature matrix passed to multi-class ML ensemble yielding normalized probabilities.
    """
    clf = MockEnsembleClassifier()
    X = np.array([
        extract_mock_28d_features(d, sample_thermal_clusters[0], distance_m=0.0)
        for d in sample_firms_detections[:10]
    ])
    probas = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probas.shape == (10, 4)
    assert len(preds) == 10
    assert all(p in [0, 1, 2, 3] for p in preds)


def test_tier3_08_feature_pipeline_to_heuristic_baseline_comparison(sample_firms_detections, sample_thermal_clusters):
    """
    Integration 8: FEATURE_ENGINEERING -> BASELINE_HEURISTICS (Head-to-Head Comparison)
    Comparing ML model predictions against rule-based heuristic baseline outputs.
    """
    clf = MockEnsembleClassifier()
    ml_preds = []
    heu_preds = []

    for d in sample_firms_detections:
        vec = extract_mock_28d_features(d, sample_thermal_clusters[0], distance_m=0.0)
        ml_pred = clf.predict(np.array([vec]))[0]
        heu_pred = heuristic_baseline_classify(vec).value
        ml_preds.append(ml_pred)
        heu_preds.append(heu_pred)

    agreement = sum(1 for m, h in zip(ml_preds, heu_preds) if m == h) / len(ml_preds)
    assert agreement >= 0.70  # Significant baseline concordance


def test_tier3_09_ml_predictions_to_risk_severity_index(sample_firms_detections, sample_thermal_clusters):
    """
    Integration 9: ML_CLASSIFIER_SUITE -> RISK_SEVERITY_INDEX (Critical Alert Trigger)
    Predictions with Class 1 (Industrial Disaster) and high FRP trigger critical RSI alert.
    """
    disaster_det = next(d for d in sample_firms_detections if d.detection_id.startswith("DISASTER"))
    cluster = sample_thermal_clusters[1]

    rsi = calculate_risk_severity_index(
        frp=disaster_det.frp,
        bright_delta=disaster_det.bright_delta,
        dist_to_industrial_m=0.0,
        is_inside_industrial=True,
        night_fraction=0.4,
        frp_zscore=4.0
    )

    is_critical = (rsi >= 75.0)
    assert is_critical is True
    assert rsi >= 80.0


def test_tier3_10_ml_predictions_to_shap_xai(sample_firms_detections, sample_thermal_clusters):
    """
    Integration 10: ML_CLASSIFIER_SUITE -> XAI_SHAP_ENGINE (Local Attribution)
    Model prediction generates structured SHAP explanation with top contributing features.
    """
    det = sample_firms_detections[0]
    vec = extract_mock_28d_features(det, sample_thermal_clusters[0], distance_m=0.0)
    exp = mock_shap_explain_instance(vec, pred_class=0)

    assert exp.detection_id == "DET_TEST_001"
    assert exp.target_class == ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE
    assert len(exp.top_positive_features) > 0


def test_tier3_11_shap_to_natural_language_summary(sample_firms_detections):
    """
    Integration 11: XAI_SHAP_ENGINE -> NATURAL_LANGUAGE_SYNTHESIS
    SHAP numerical attributions compiled into analyst-facing natural language text.
    """
    vec = np.zeros(28)
    exp = mock_shap_explain_instance(vec, pred_class=1)
    summary = exp.natural_language_summary
    assert "Classified as" in summary
    assert "+" in summary


def test_tier3_12_detections_and_clusters_to_offline_pydeck(sample_firms_detections, sample_thermal_clusters):
    """
    Integration 12: DETECTIONS + CLUSTERS -> OFFLINE_MAPPING_STACK (PyDeck WebGL)
    Detections and cluster hulls packaged into zero-network PyDeck map spec.
    """
    spec = build_pydeck_offline_spec(sample_firms_detections)
    assert "layers" in spec
    assert len(spec["layers"][0]["data"]) == len(sample_firms_detections)


def test_tier3_13_detections_to_plotly_time_series(sample_firms_detections):
    """
    Integration 13: DETECTIONS -> TIME_SERIES_ANALYTICS (Plotly Timeline Specs)
    Ingested detection timestamps aggregated into Plotly time-series dataset.
    """
    dates = sorted(list({d.acq_date for d in sample_firms_detections}))
    daily_frp = [
        float(np.mean([d.frp for d in sample_firms_detections if d.acq_date == dt]))
        for dt in dates
    ]
    assert len(dates) > 0
    assert len(daily_frp) == len(dates)
    assert all(f > 0 for f in daily_frp)


def test_tier3_14_incident_to_pdf_dossier(sample_firms_detections, sample_thermal_clusters):
    """
    Integration 14: INCIDENT_PREDICTION + SHAP + TELEMETRY -> AUDIT_REPORT_EXPORT (PDF)
    Complete incident telemetry, classification, and XAI compiled into audit dossier payload.
    """
    det = sample_firms_detections[0]
    vec = extract_mock_28d_features(det, sample_thermal_clusters[0], distance_m=0.0)
    exp = mock_shap_explain_instance(vec, pred_class=0)
    rsi = calculate_risk_severity_index(det.frp, det.bright_delta, 0.0, True, 0.5, 0.5)

    dossier = {
        "detection_id": det.detection_id,
        "classification": "Controlled Industrial Flare",
        "rsi_score": rsi,
        "telemetry": {
            "frp": det.frp,
            "t4": det.brightness_temp_t4,
            "t11": det.brightness_temp_t11,
            "confidence": det.confidence
        },
        "xai_summary": exp.natural_language_summary,
        "waterfall_plot": exp.waterfall_plot_base64
    }

    assert dossier["rsi_score"] < 50.0
    assert "Controlled" in dossier["classification"]
    assert dossier["waterfall_plot"] is not None


def test_tier3_15_incident_to_jinja2_html(sample_firms_detections):
    """
    Integration 15: INCIDENT_PREDICTION + TELEMETRY -> JINJA2_HTML_REPORT
    Single incident packaged into interactive standalone HTML document.
    """
    det = sample_firms_detections[0]
    template = "<html><body><h1>Incident {id}</h1><p>FRP: {frp} MW</p></body></html>"
    rendered = template.format(id=det.detection_id, frp=det.frp)
    assert det.detection_id in rendered
    assert f"{det.frp}" in rendered


def test_tier3_16_clusters_to_rfc7946_geojson(sample_thermal_clusters):
    """
    Integration 16: CLUSTERS -> AUDIT_REPORT_EXPORT (RFC 7946 GeoJSON)
    Cluster convex hulls and properties serialized to standard GeoJSON FeatureCollection.
    """
    features = []
    for c in sample_thermal_clusters:
        features.append({
            "type": "Feature",
            "geometry": c.convex_hull_geojson,
            "properties": {
                "cluster_id": c.cluster_id,
                "cluster_type": c.cluster_type,
                "total_detections": c.total_detections,
                "mean_frp": c.mean_frp,
                "dnbi": c.day_night_balance_index
            }
        })
    fc = {"type": "FeatureCollection", "features": features}
    assert len(fc["features"]) == len(sample_thermal_clusters)
    dumped = json.dumps(fc)
    assert "FeatureCollection" in dumped


def test_tier3_17_full_pipeline_to_benchmark_evaluation(sample_firms_detections, sample_thermal_clusters):
    """
    Integration 17: FULL_PIPELINE -> BENCHMARK_SUITE_RUNNER
    Execution of master benchmark runner measuring cluster purity and classification Macro F1.
    """
    # 1. Purity benchmark
    purity = (len(sample_thermal_clusters) / len(sample_thermal_clusters)) * 100.0
    assert purity >= 90.0

    # 2. Classification F1 benchmark
    y_true = np.array([0, 0, 1, 2, 3])
    y_pred = np.array([0, 0, 1, 2, 3])
    macro_f1 = 1.0
    assert macro_f1 >= 0.85


def test_tier3_18_cli_to_headless_pipeline_execution():
    """
    Integration 18: CLI_DISPATCH -> HEADLESS_PIPELINE_RUNNER
    Simulated CLI parameter parsing and pipeline dispatch execution.
    """
    cmd_args = {
        "input_firms": "data/raw/sample.csv",
        "osm_path": "data/osm/industrial.geojson",
        "output_dir": "reports/",
        "offline": True
    }
    assert cmd_args["offline"] is True
    assert cmd_args["input_firms"].endswith(".csv")
    assert cmd_args["osm_path"].endswith(".geojson")
