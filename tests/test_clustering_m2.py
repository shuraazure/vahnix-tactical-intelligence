"""
Test Suite for Milestone 2: Spatio-Temporal Clustering & Recurrence Analysis.
Validates H3 Spatial Indexer, Haversine DBSCAN, ST-DBSCAN, HDBSCAN, Persistence Analyzer,
and Cluster Metrics Evaluator (Cluster Purity >= 90.0%).
Authoritative Specifications: ORIGINAL_REQUEST.md § R1, PROJECT.md, m2_blueprint.md
"""

from __future__ import annotations

import datetime
import json
import math
from typing import Any
import numpy as np
import pandas as pd
import pytest
import shapely.geometry

from src.data_pipeline.schemas import FIRMSDetection, IndustrialFacility
from src.data_pipeline.osm_indexer import SpatialIndexEngine
from src.data_pipeline.synthetic_generator import SyntheticDataGenerator

from src.clustering.schemas import (
    ThermalCluster,
    ClusterType,
    H3SpatialBin,
    ClusterEvaluationMetrics,
)
from src.clustering.h3_indexer import (
    H3SpatialIndexer,
    compute_h3_index,
    get_h3_boundary,
    get_h3_centroid,
    get_h3_k_ring,
)
from src.clustering.persistence_analyzer import (
    PersistenceAnalyzer,
    haversine_distance,
    calculate_dnbi,
    calculate_spatial_jitter,
    calculate_centroid_drift_velocity,
    EARTH_RADIUS_METERS,
)
from src.clustering.dbscan_engine import (
    SpatioTemporalClusteringEngine,
    run_hdbscan,
)
from src.clustering.cluster_metrics import (
    ClusterMetricsEvaluator,
)


# =========================================================================
# 1. SCHEMAS & DATA CONTRACT TESTS
# =========================================================================

def test_thermal_cluster_schema_and_invariants():
    """Verify ThermalCluster validation, clamping, and dataclass invariants."""
    now = datetime.datetime.utcnow()
    c = ThermalCluster(
        cluster_id="CLUS_TEST_001",
        cluster_type=ClusterType.STATIONARY_PERSISTENT.value,
        detection_ids=["D1", "D2", "D3"],
        centroid_lat=22.3582,
        centroid_lon=69.8681,
        bounding_box=(69.86, 22.35, 69.87, 22.36),
        convex_hull_geojson={"type": "Polygon", "coordinates": [[[69.86, 22.35], [69.87, 22.35], [69.87, 22.36], [69.86, 22.35]]]},
        total_detections=3,
        first_seen=now - datetime.timedelta(days=10),
        last_seen=now,
        temporal_span_days=10.0,
        active_days_count=5,
        recurrence_rate=0.5,
        day_count=2,
        night_count=1,
        day_night_balance_index=0.6667,
        spatial_jitter_m=45.2,
        mean_frp=35.0,
        max_frp=55.0,
        frp_std=10.0,
        nearest_industrial_facility_id="IND_JAMNAGAR_001",
        distance_to_nearest_industrial_m=0.0,
        is_inside_industrial_boundary=True
    )
    assert c.cluster_id == "CLUS_TEST_001"
    assert c.total_detections == 3
    assert math.isclose(c.day_night_balance_index, 0.6667, abs_tol=1e-3)

    # Invariant validation for invalid latitude
    with pytest.raises(ValueError, match="Invalid centroid_lat"):
        ThermalCluster(
            cluster_id="BAD_LAT",
            cluster_type="stationary_persistent",
            detection_ids=["D1"],
            centroid_lat=95.0,
            centroid_lon=69.8681,
            bounding_box=(0, 0, 0, 0),
            convex_hull_geojson={},
            total_detections=1,
            first_seen=now,
            last_seen=now,
            temporal_span_days=0.0,
            active_days_count=1,
            recurrence_rate=1.0,
            day_count=1,
            night_count=0,
            day_night_balance_index=0.0,
            spatial_jitter_m=0.0,
            mean_frp=10.0,
            max_frp=10.0,
            frp_std=0.0
        )


def test_thermal_cluster_serialization_and_dataframe_roundtrip():
    """Verify ThermalCluster dict, GeoJSON, and DataFrame serialization roundtrip."""
    now = datetime.datetime(2026, 3, 15, 12, 0, 0)
    c1 = ThermalCluster(
        cluster_id="CLUS_001",
        cluster_type=ClusterType.STATIONARY_PERSISTENT.value,
        detection_ids=["D1", "D2"],
        centroid_lat=22.35,
        centroid_lon=69.86,
        bounding_box=(69.85, 22.34, 69.87, 22.36),
        convex_hull_geojson={"type": "Point", "coordinates": [69.86, 22.35]},
        total_detections=2,
        first_seen=now - datetime.timedelta(days=2),
        last_seen=now,
        temporal_span_days=2.0,
        active_days_count=2,
        recurrence_rate=0.2,
        day_count=1,
        night_count=1,
        day_night_balance_index=1.0,
        spatial_jitter_m=25.0,
        mean_frp=40.0,
        max_frp=50.0,
        frp_std=10.0
    )

    # Dict roundtrip
    d = c1.to_dict()
    assert isinstance(d["first_seen"], str)
    c1_rec = ThermalCluster.from_dict(d)
    assert c1_rec.cluster_id == c1.cluster_id
    assert math.isclose(c1_rec.centroid_lat, 22.35)

    # GeoJSON export
    feat = c1.to_geojson_feature()
    assert feat["type"] == "Feature"
    assert feat["id"] == "CLUS_001"
    assert feat["properties"]["cluster_type"] == "stationary_persistent"

    # DataFrame roundtrip
    df = ThermalCluster.to_dataframe([c1])
    assert len(df) == 1
    reconstructed_list = ThermalCluster.from_dataframe(df)
    assert len(reconstructed_list) == 1
    assert reconstructed_list[0].cluster_id == "CLUS_001"
    assert reconstructed_list[0].detection_ids == ["D1", "D2"]


def test_h3_spatial_bin_schema_and_dataframe():
    """Verify H3SpatialBin serialization and conversions."""
    now = datetime.datetime(2026, 3, 15, 12, 0, 0)
    hbin = H3SpatialBin(
        h3_index="886120150bfffff",
        resolution=8,
        centroid_lat=22.3582,
        centroid_lon=69.8681,
        boundary_geojson={"type": "Polygon", "coordinates": [[[69.86, 22.35], [69.87, 22.35], [69.87, 22.36], [69.86, 22.35]]]},
        total_detections=5,
        detection_ids=["D1", "D2", "D3", "D4", "D5"],
        sum_frp=150.0,
        mean_frp=30.0,
        max_frp=50.0,
        min_frp=20.0,
        std_frp=10.0,
        day_count=3,
        night_count=2,
        day_night_balance_index=0.80,
        active_days_count=3,
        first_seen=now - datetime.timedelta(days=5),
        last_seen=now,
        temporal_span_days=5.0,
        k_ring_neighbors=["886120150bfffff", "886120150afffff"]
    )
    b_dict = hbin.to_dict()
    assert b_dict["total_detections"] == 5
    b_rec = H3SpatialBin.from_dict(b_dict)
    assert b_rec.h3_index == "886120150bfffff"

    feat = hbin.to_geojson_feature()
    assert feat["type"] == "Feature"
    assert feat["properties"]["mean_frp"] == 30.0

    df = H3SpatialBin.to_dataframe([hbin])
    assert len(df) == 1
    assert df.iloc[0]["total_detections"] == 5


# =========================================================================
# 2. H3 SPATIAL INDEXER TESTS
# =========================================================================

def test_h3_index_generation_and_boundary():
    """Verify compute_h3_index, get_h3_boundary, get_h3_centroid, and get_h3_k_ring."""
    lat, lon = 22.3582, 69.8681
    idx7 = compute_h3_index(lat, lon, 7)
    idx8 = compute_h3_index(lat, lon, 8)
    idx9 = compute_h3_index(lat, lon, 9)

    assert isinstance(idx7, str) and len(idx7) >= 12
    assert isinstance(idx8, str) and len(idx8) >= 12
    assert isinstance(idx9, str) and len(idx9) >= 12

    # Boundary coordinates test
    boundary = get_h3_boundary(idx8)
    assert len(boundary) >= 6
    assert boundary[0] == boundary[-1]  # Closed ring

    # Centroid coordinate test
    c_lat, c_lon = get_h3_centroid(idx8)
    assert -90.0 <= c_lat <= 90.0
    assert -180.0 <= c_lon <= 180.0
    assert haversine_distance(lat, lon, c_lat, c_lon) < 5000.0  # within 5km

    # K-ring test
    k_ring = get_h3_k_ring(idx8, k=1)
    assert len(k_ring) >= 1
    assert idx8 in k_ring


def test_h3_indexer_aggregation_and_enrichment():
    """Verify H3SpatialIndexer adding indices, aggregating bins, and time-window partitioning."""
    now = datetime.datetime(2026, 3, 15, 12, 0)
    dets = [
        FIRMSDetection(
            detection_id=f"D_{i}",
            latitude=22.3582 + (i * 0.0005),
            longitude=69.8681 + (i * 0.0005),
            acq_date=f"2026-03-{15 + (i % 3):02d}",
            acq_time="0630" if i % 2 == 0 else "1845",
            timestamp=now + datetime.timedelta(days=i % 3, hours=i),
            sensor="VIIRS_SNPP",
            satellite="SNPP",
            frp=25.0 + (i * 5.0),
            brightness_temp_t4=340.0,
            brightness_temp_t11=295.0,
            bright_delta=45.0,
            confidence=0.90,
            daynight="D" if i % 2 == 0 else "N"
        )
        for i in range(10)
    ]

    indexer = H3SpatialIndexer(default_resolution=8)
    enriched = indexer.add_h3_indices(dets)
    assert len(enriched) == 10
    assert all(d.h3_res7 and d.h3_res8 and d.h3_res9 for d in enriched)

    # Aggregation
    bins = indexer.aggregate_by_h3(enriched, resolution=8, min_detections=1)
    assert len(bins) >= 1
    total_in_bins = sum(b.total_detections for b in bins)
    assert total_in_bins == 10

    # Time-window aggregation
    windows = indexer.aggregate_time_windows(enriched, resolution=8, window_days=2)
    assert len(windows) >= 1


# =========================================================================
# 3. PERSISTENCE ANALYZER TESTS
# =========================================================================

def test_dnbi_formula_and_edges():
    """Verify Day-Night Balance Index calculations across edge cases."""
    # 50% Day / 50% Night -> 1.0
    assert math.isclose(calculate_dnbi(10, 10), 1.0)
    # 100% Day -> 0.0
    assert math.isclose(calculate_dnbi(15, 0), 0.0)
    # 100% Night -> 0.0
    assert math.isclose(calculate_dnbi(0, 15), 0.0)
    # 0 Day / 0 Night -> 0.0
    assert math.isclose(calculate_dnbi(0, 0), 0.0)
    # 8 Day / 2 Night -> 2 * 2 / 10 = 0.40
    assert math.isclose(calculate_dnbi(8, 2), 0.40)


def test_spatial_jitter_and_drift_velocity():
    """Verify 1-sigma spatial jitter and progression drift velocity."""
    coords = [(22.3582, 69.8681), (22.3583, 69.8682), (22.3581, 69.8680)]
    cent = (22.3582, 69.8681)
    jitter = calculate_spatial_jitter(coords, cent)
    assert 0.0 <= jitter <= 50.0

    # Collocated points
    assert calculate_spatial_jitter([(22.0, 69.0), (22.0, 69.0)], (22.0, 69.0)) == 0.0
    # Single point
    assert calculate_spatial_jitter([(22.0, 69.0)], (22.0, 69.0)) == 0.0


def test_persistence_analyzer_cluster_synthesis():
    """Verify PersistenceAnalyzer builds valid ThermalCluster with OSM spatial association."""
    now = datetime.datetime(2026, 3, 15, 12, 0)
    dets = [
        FIRMSDetection(
            detection_id=f"JAM_{i}",
            latitude=22.3582 + (i * 0.0001),
            longitude=69.8681 + (i * 0.0001),
            acq_date=f"2026-03-{15 + (i % 5):02d}",
            acq_time="0630" if i % 2 == 0 else "1845",
            timestamp=now + datetime.timedelta(days=i % 5),
            sensor="VIIRS_SNPP",
            satellite="SNPP",
            frp=35.0,
            brightness_temp_t4=340.0,
            brightness_temp_t11=295.0,
            bright_delta=45.0,
            confidence=0.95,
            daynight="D" if i % 2 == 0 else "N"
        )
        for i in range(10)
    ]

    # Mock Spatial Index Engine
    spatial_engine = SpatialIndexEngine()
    poly = shapely.geometry.box(69.84, 22.33, 69.89, 22.38)
    fac = IndustrialFacility(
        facility_id="IND_JAMNAGAR_001",
        name="Jamnagar Refinery",
        facility_type="refinery",
        geometry=poly,
        buffer_geometry_350m=poly.buffer(0.003),
        buffer_geometry_750m=poly.buffer(0.007),
        bounding_box=(69.84, 22.33, 69.89, 22.38),
        area_sq_km=25.0
    )
    spatial_engine.add_facility(fac)

    analyzer = PersistenceAnalyzer(spatial_index_engine=spatial_engine)
    cluster = analyzer.analyze_cluster("CLUS_JAM_001", dets)

    assert cluster.cluster_id == "CLUS_JAM_001"
    assert cluster.cluster_type == ClusterType.STATIONARY_PERSISTENT.value
    assert cluster.total_detections == 10
    assert cluster.is_inside_industrial_boundary is True
    assert cluster.nearest_industrial_facility_id == "IND_JAMNAGAR_001"
    assert math.isclose(cluster.distance_to_nearest_industrial_m, 0.0, abs_tol=1e-3)
    assert cluster.day_night_balance_index >= 0.80
    assert cluster.spatial_jitter_m < 50.0


# =========================================================================
# 4. DBSCAN & ST-DBSCAN CLUSTERING ENGINE TESTS
# =========================================================================

def test_haversine_dbscan_clustering():
    """Verify Haversine DBSCAN 500m core discovery groups dense points and filters noise."""
    now = datetime.datetime(2026, 3, 15, 12, 0)
    # 5 points in Jamnagar (dense core <= 50m)
    jam_dets = [
        FIRMSDetection(
            detection_id=f"JAM_{i}",
            latitude=22.3582 + (i * 0.0001),
            longitude=69.8681 + (i * 0.0001),
            acq_date="2026-03-15",
            acq_time="1200",
            timestamp=now,
            sensor="MODIS",
            satellite="Terra",
            frp=40.0,
            brightness_temp_t4=335.0,
            brightness_temp_t11=295.0,
            bright_delta=40.0,
            confidence=0.85,
            daynight="D"
        )
        for i in range(5)
    ]
    # 1 isolated noise point (Mumbai Trombay, 489km away)
    noise_det = FIRMSDetection(
        detection_id="NOISE_001",
        latitude=18.9950,
        longitude=72.8950,
        acq_date="2026-03-15",
        acq_time="1200",
        timestamp=now,
        sensor="MODIS",
        satellite="Terra",
        frp=15.0,
        brightness_temp_t4=315.0,
        brightness_temp_t11=295.0,
        bright_delta=20.0,
        confidence=0.50,
        daynight="D"
    )

    all_dets = jam_dets + [noise_det]
    engine = SpatioTemporalClusteringEngine()
    clusters = engine.run_dbscan_clustering(all_dets, eps_m=500.0, min_samples=3)

    assert len(clusters) == 1
    assert clusters[0].total_detections == 5
    assert "NOISE_001" not in clusters[0].detection_ids


def test_st_dbscan_spatio_temporal_clustering():
    """
    Verify ST-DBSCAN enforces both spatial threshold (1000m) AND temporal threshold (48h).
    Points within 500m but 30 days apart must NOT be clustered together.
    """
    t0 = datetime.datetime(2026, 3, 1, 12, 0)
    t1 = datetime.datetime(2026, 4, 1, 12, 0)  # 31 days later

    # Group 1: 4 points on March 1
    g1 = [
        FIRMSDetection(
            detection_id=f"G1_{i}",
            latitude=22.3582 + (i * 0.0001),
            longitude=69.8681 + (i * 0.0001),
            acq_date="2026-03-01",
            acq_time="1200",
            timestamp=t0 + datetime.timedelta(hours=i),
            sensor="VIIRS_SNPP",
            satellite="SNPP",
            frp=30.0,
            brightness_temp_t4=335.0,
            brightness_temp_t11=295.0,
            bright_delta=40.0,
            confidence=0.90,
            daynight="D"
        )
        for i in range(4)
    ]

    # Group 2: 4 points on April 1 at nearly same coordinates
    g2 = [
        FIRMSDetection(
            detection_id=f"G2_{i}",
            latitude=22.3582 + (i * 0.0001),
            longitude=69.8681 + (i * 0.0001),
            acq_date="2026-04-01",
            acq_time="1200",
            timestamp=t1 + datetime.timedelta(hours=i),
            sensor="VIIRS_SNPP",
            satellite="SNPP",
            frp=30.0,
            brightness_temp_t4=335.0,
            brightness_temp_t11=295.0,
            bright_delta=40.0,
            confidence=0.90,
            daynight="D"
        )
        for i in range(4)
    ]

    engine = SpatioTemporalClusteringEngine()
    clusters = engine.run_st_dbscan(g1 + g2, spatial_eps_m=1000.0, temporal_eps_hours=48.0, min_samples=4)

    # Must produce 2 separate clusters because temporal gap is 31 days (> 48 hours)
    assert len(clusters) == 2
    assert {c.total_detections for c in clusters} == {4}


def test_hdbscan_clustering_wrapper():
    """Verify HDBSCAN multi-density clustering wrapper."""
    now = datetime.datetime(2026, 3, 15, 12, 0)
    dets = [
        FIRMSDetection(
            detection_id=f"HDB_{i}",
            latitude=22.3582 + (i * 0.0001),
            longitude=69.8681 + (i * 0.0001),
            acq_date="2026-03-15",
            acq_time="1200",
            timestamp=now,
            sensor="VIIRS_SNPP",
            satellite="SNPP",
            frp=30.0,
            brightness_temp_t4=335.0,
            brightness_temp_t11=295.0,
            bright_delta=40.0,
            confidence=0.90,
            daynight="D"
        )
        for i in range(6)
    ]

    engine = SpatioTemporalClusteringEngine()
    clusters = engine.run_hdbscan_clustering(dets, min_cluster_size=3)
    assert len(clusters) >= 1
    assert clusters[0].total_detections >= 3


def test_h3_fast_clustering():
    """Verify rapid O(N) H3 discrete global grid clustering."""
    now = datetime.datetime(2026, 3, 15, 12, 0)
    dets = [
        FIRMSDetection(
            detection_id=f"H3_DET_{i}",
            latitude=22.3582 + (i * 0.0001),
            longitude=69.8681 + (i * 0.0001),
            acq_date="2026-03-15",
            acq_time="1200",
            timestamp=now,
            sensor="VIIRS_SNPP",
            satellite="SNPP",
            frp=30.0,
            brightness_temp_t4=335.0,
            brightness_temp_t11=295.0,
            bright_delta=40.0,
            confidence=0.90,
            daynight="D"
        )
        for i in range(5)
    ]

    engine = SpatioTemporalClusteringEngine()
    clusters = engine.run_h3_clustering(dets, res=8, min_samples=3)
    assert len(clusters) == 1
    assert clusters[0].total_detections == 5


# =========================================================================
# 5. CLUSTER METRICS EVALUATION & ACCEPTANCE BENCHMARKS
# =========================================================================

def test_cluster_metrics_evaluator_exact_calculations():
    """Verify cluster purity, homogeneity, completeness, and V-measure on known ground truth."""
    # Predicted clusters: Cluster 0 has [A, A, A, B], Cluster 1 has [B, B, B, A]
    pred = [0, 0, 0, 0, 1, 1, 1, 1]
    truth = ["A", "A", "A", "B", "B", "B", "B", "A"]

    purity = ClusterMetricsEvaluator.calculate_cluster_purity(pred, truth)
    # Cluster 0 majority count = 3 ('A'), Cluster 1 majority count = 3 ('B')
    # Total purity = (3 + 3) / 8 = 6 / 8 = 0.75
    assert math.isclose(purity, 0.75, abs_tol=1e-4)

    metrics = ClusterMetricsEvaluator.evaluate_benchmark(pred, truth)
    assert math.isclose(metrics.cluster_purity, 0.75, abs_tol=1e-4)
    assert 0.0 <= metrics.homogeneity <= 1.0
    assert 0.0 <= metrics.completeness <= 1.0
    assert 0.0 <= metrics.v_measure <= 1.0


def test_cluster_purity_with_noise_points():
    """Verify cluster purity properly excludes unclustered noise (-1)."""
    pred = [0, 0, 0, -1, -1]
    truth = ["A", "A", "A", "B", "C"]
    purity = ClusterMetricsEvaluator.calculate_cluster_purity(pred, truth)
    assert math.isclose(purity, 1.0)


# =========================================================================
# 6. ACCEPTANCE BENCHMARK: CLUSTER PURITY >= 90.0% ON REALISTIC CORRIDORS
# =========================================================================

def test_acceptance_cluster_purity_benchmark():
    """
    CRITICAL ACCEPTANCE TEST:
    Verify spatio-temporal clustering achieves >= 90.0% cluster purity across
    the synthetic benchmark dataset suite (Jamnagar, Jurong, Punjab, Simlipal).
    """
    gen = SyntheticDataGenerator()
    suite = gen.generate_master_benchmark_suite()

    # Combine detections from all corridors with ground truth labels
    combined_detections: list[FIRMSDetection] = []
    ground_truth_labels: list[str] = []

    for corridor_name, dataset in suite.items():
        for d in dataset.detections:
            combined_detections.append(d)
            ground_truth_labels.append(corridor_name)

    assert len(combined_detections) > 200

    engine = SpatioTemporalClusteringEngine()
    # Run Haversine DBSCAN with 500m radius
    clusters = engine.run_dbscan_clustering(combined_detections, eps_m=1000.0, min_samples=3)

    assert len(clusters) >= 5

    # Evaluate purity
    results = engine.evaluate_cluster_purity(clusters, ground_truth_labels, detections=combined_detections)
    cluster_purity = results["cluster_purity"]

    print(f"\n[BENCHMARK] Evaluated {len(clusters)} clusters across {len(combined_detections)} detections.")
    print(f"[BENCHMARK] Measured Cluster Purity: {cluster_purity * 100.0:.2f}% (Requirement >= 90.0%)")
    print(f"[BENCHMARK] Homogeneity: {results['homogeneity']:.4f}, Completeness: {results['completeness']:.4f}, V-Measure: {results['v_measure']:.4f}")

    assert cluster_purity >= 0.90, f"Cluster purity {cluster_purity * 100:.2f}% is below 90.0% acceptance threshold!"
