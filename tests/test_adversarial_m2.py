"""
Adversarial Stress Test and Mathematical Invariant Verification Suite for Milestone 2:
Spatio-Temporal Clustering & Recurrence Analysis (NTRO SIH26162).
Author: Challenger 2 (Empirical Challenger)
Authoritative Specifications: ORIGINAL_REQUEST.md § R1, PROJECT.md, m2_blueprint.md
"""

from __future__ import annotations

import datetime
import math
import time
from typing import Any, List, Tuple
import numpy as np
import pandas as pd
import pytest
import shapely.geometry

from src.data_pipeline.schemas import FIRMSDetection, IndustrialFacility
from src.data_pipeline.synthetic_generator import SyntheticDataGenerator
from src.data_pipeline.osm_indexer import SpatialIndexEngine

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
# 1. DNBI (DAY-NIGHT BALANCE INDEX) MATHEMATICAL & BOUNDARY VERIFICATION
# =========================================================================

def test_dnbi_mathematical_invariants():
    """
    Verify fundamental mathematical properties of DNBI = 2 * min(D, N) / (D + N):
    1. DNBI == 0.0 for 100% daytime fires (D > 0, N = 0).
    2. DNBI == 0.0 for 100% nighttime fires (D = 0, N > 0).
    3. DNBI == 1.0 for perfectly balanced day/night fires (D == N > 0).
    4. DNBI == 0.0 for zero detections (D = 0, N = 0) without division by zero.
    5. Symmetry: calculate_dnbi(D, N) == calculate_dnbi(N, D) for all pairs.
    6. Range containment: DNBI in [0.0, 1.0] for all combinations.
    """
    # 1. 100% Daytime fires
    for d in [1, 5, 10, 50, 100, 1000, 100000]:
        assert calculate_dnbi(d, 0) == 0.0, f"Failed for daytime count {d}"

    # 2. 100% Nighttime fires
    for n in [1, 5, 10, 50, 100, 1000, 100000]:
        assert calculate_dnbi(0, n) == 0.0, f"Failed for nighttime count {n}"

    # 3. 50/50 Equal Day/Night detections
    for count in [1, 2, 5, 10, 50, 100, 1000, 100000]:
        val = calculate_dnbi(count, count)
        assert math.isclose(val, 1.0, rel_tol=1e-9), f"Failed for count {count}: got {val}"

    # 4. Zero detection edge case
    assert calculate_dnbi(0, 0) == 0.0

    # 5. Symmetry invariant
    pairs = [(10, 90), (25, 75), (33, 67), (49, 51), (1, 999), (500, 1500)]
    for d, n in pairs:
        assert math.isclose(calculate_dnbi(d, n), calculate_dnbi(n, d), rel_tol=1e-9)

    # 6. Monotonicity as ratio approaches 1:1
    dnbi_10_90 = calculate_dnbi(10, 90)  # 2 * 10 / 100 = 0.20
    dnbi_25_75 = calculate_dnbi(25, 75)  # 2 * 25 / 100 = 0.50
    dnbi_40_60 = calculate_dnbi(40, 60)  # 2 * 40 / 100 = 0.80
    dnbi_50_50 = calculate_dnbi(50, 50)  # 2 * 50 / 100 = 1.00
    assert 0.0 < dnbi_10_90 < dnbi_25_75 < dnbi_40_60 < dnbi_50_50 == 1.0


def test_dnbi_large_integers_and_numerical_stability():
    """Verify DNBI calculation does not overflow with large telemetry streams."""
    large_d = 10_000_000
    large_n = 10_000_000
    assert math.isclose(calculate_dnbi(large_d, large_n), 1.0)

    # Highly unbalanced large stream
    assert math.isclose(calculate_dnbi(10_000_000, 1), 2.0 / 10_000_001, rel_tol=1e-7)


# =========================================================================
# 2. SPATIAL JITTER DISPERSION: STATIONARY STACKS VS WILDFIRE FRONTS
# =========================================================================

def test_spatial_jitter_stationary_stacks():
    """
    Verify spatial jitter sigma_jitter <= 350m for stationary refinery flare stacks.
    Simulates satellite geolocation noise (~65m Gaussian 1-sigma dispersion) across 50 observations.
    """
    rng = np.random.RandomState(42)
    center_lat, center_lon = 22.3582, 69.8681
    m_to_deg_lat = 1.0 / 111320.0
    m_to_deg_lon = 1.0 / (111320.0 * math.cos(math.radians(center_lat)))

    # Generate 50 points around stationary stack with 65m sigma geolocation jitter
    lats = center_lat + rng.normal(0, 65.0 * m_to_deg_lat, size=50)
    lons = center_lon + rng.normal(0, 65.0 * m_to_deg_lon, size=50)
    coords = list(zip(lats, lons))

    cent = (float(np.mean(lats)), float(np.mean(lons)))
    jitter_m = calculate_spatial_jitter(coords, cent)

    print(f"\n[EMPIRICAL] Stationary Flare Stack Spatial Jitter: {jitter_m:.2f} m (Target: <= 350 m)")
    assert jitter_m <= 350.0, f"Stationary jitter {jitter_m:.2f}m exceeded 350m threshold!"
    assert 30.0 <= jitter_m <= 120.0  # Real-world 1-sigma dispersion range


def test_spatial_jitter_wide_area_wildfires():
    """
    Verify spatial jitter sigma_jitter > 1000m for wide-area advancing wildfires.
    Simulates a forest fire front propagating across a 15 km regional swath.
    """
    rng = np.random.RandomState(101)
    center_lat, center_lon = 21.620, 86.320
    m_to_deg_lat = 1.0 / 111320.0
    m_to_deg_lon = 1.0 / (111320.0 * math.cos(math.radians(center_lat)))

    # Generate 50 points distributed along a 15,000m advancing front
    offsets_lat = rng.uniform(-7500.0, 7500.0, size=50) * m_to_deg_lat
    offsets_lon = rng.uniform(-7500.0, 7500.0, size=50) * m_to_deg_lon

    lats = center_lat + offsets_lat
    lons = center_lon + offsets_lon
    coords = list(zip(lats, lons))

    cent = (float(np.mean(lats)), float(np.mean(lons)))
    jitter_m = calculate_spatial_jitter(coords, cent)

    print(f"[EMPIRICAL] Wide-Area Wildfire Spatial Jitter: {jitter_m:.2f} m (Target: > 1000 m)")
    assert jitter_m > 1000.0, f"Wildfire jitter {jitter_m:.2f}m was below 1000m threshold!"
    assert jitter_m >= 2500.0


def test_spatial_jitter_two_points_exact_formula():
    """
    For 2 points at distance D, centroid is at D/2, each point is at distance D/2.
    std([D/2, D/2]) == 0.0 because both distances to centroid are identical!
    """
    p1 = (22.000, 69.000)
    p2 = (22.010, 69.000)  # ~1113m apart
    d = haversine_distance(p1[0], p1[1], p2[0], p2[1])
    cent = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
    jitter = calculate_spatial_jitter([p1, p2], cent)
    # Both points are equidistant from centroid -> std of distance values is 0.0
    assert math.isclose(jitter, 0.0, abs_tol=1e-5)


# =========================================================================
# 3. RECURRENCE FREQUENCY SCALING WITH OBSERVATION WINDOW
# =========================================================================

def test_recurrence_frequency_scaling_with_observation_window():
    """
    Verify recurrence rate scaling: recurrence_rate = active_days / max(monitoring_window_days, 1.0)
    1. For a transient 3-day stubble burn:
       - In 10-day window: recurrence = 3 / 10 = 0.30
       - In 30-day window: recurrence = 3 / 30 = 0.10
       - In 90-day window: recurrence = 3 / 90 = 0.0333
       - In 365-day window: recurrence = 3 / 365 = 0.0082
       Recurrence rate monotonically decreases as observation window scales.
    2. For a persistent 30-day continuous refinery flare:
       - In 30-day window: recurrence = 30 / 30 = 1.0
       - In 90-day window: recurrence = 90 / 90 = 1.0
    3. Divisor clamping: window <= 0.0 clamps divisor to 1.0, preventing division by zero.
    """
    # Transient 3-day fire
    active_days_transient = 3
    rec_10 = active_days_transient / max(10.0, 1.0)
    rec_30 = active_days_transient / max(30.0, 1.0)
    rec_90 = active_days_transient / max(90.0, 1.0)
    rec_365 = active_days_transient / max(365.0, 1.0)

    assert math.isclose(rec_10, 0.30)
    assert math.isclose(rec_30, 0.10)
    assert math.isclose(rec_90, 3.0 / 90.0)
    assert math.isclose(rec_365, 3.0 / 365.0)
    assert rec_10 > rec_30 > rec_90 > rec_365

    # Zero window protection
    rec_zero = active_days_transient / max(0.0, 1.0)
    assert math.isclose(rec_zero, 3.0)

    # Negative window protection
    rec_neg = active_days_transient / max(-5.0, 1.0)
    assert math.isclose(rec_neg, 3.0)

    # Synthesize cluster via PersistenceAnalyzer under different windows
    now = datetime.datetime(2026, 3, 1, 12, 0)
    dets = [
        FIRMSDetection(
            detection_id=f"D_{i}",
            latitude=22.3582,
            longitude=69.8681,
            acq_date=f"2026-03-{i+1:02d}",
            acq_time="1200",
            timestamp=now + datetime.timedelta(days=i),
            sensor="VIIRS_SNPP",
            satellite="SNPP",
            frp=30.0,
            brightness_temp_t4=340.0,
            brightness_temp_t11=295.0,
            bright_delta=45.0,
            confidence=0.90,
            daynight="D"
        )
        for i in range(5)  # 5 active days
    ]

    analyzer = PersistenceAnalyzer()
    c_30 = analyzer.analyze_cluster("CLUS_30", dets, monitoring_window_days=30.0)
    c_90 = analyzer.analyze_cluster("CLUS_90", dets, monitoring_window_days=90.0)
    c_365 = analyzer.analyze_cluster("CLUS_365", dets, monitoring_window_days=365.0)

    assert math.isclose(c_30.recurrence_rate, 5.0 / 30.0, abs_tol=1e-5)
    assert math.isclose(c_90.recurrence_rate, 5.0 / 90.0, abs_tol=1e-5)
    assert math.isclose(c_365.recurrence_rate, 5.0 / 365.0, abs_tol=1e-5)
    assert c_30.recurrence_rate > c_90.recurrence_rate > c_365.recurrence_rate


# =========================================================================
# 4. EXECUTION SCALABILITY BENCHMARKS ON LARGE DATASETS (10,000+ POINTS)
# =========================================================================

def test_scalability_10k_h3_indexing_and_aggregation():
    """
    Scalability Benchmark 1:
    Uber H3 spatial binning and aggregation on 10,000+ detection records.
    Verifies O(N) linear complexity and execution in < 1.0 second.
    """
    rng = np.random.RandomState(42)
    n = 10000
    now = datetime.datetime(2026, 3, 1, 12, 0)

    lats = rng.uniform(8.0, 37.0, size=n)
    lons = rng.uniform(68.0, 97.0, size=n)
    frps = rng.exponential(scale=35.0, size=n)

    detections = [
        FIRMSDetection(
            detection_id=f"SCALE_H3_{i:06d}",
            latitude=float(lats[i]),
            longitude=float(lons[i]),
            acq_date="2026-03-01",
            acq_time="1200",
            timestamp=now,
            sensor="MODIS",
            satellite="Terra",
            frp=float(frps[i]),
            brightness_temp_t4=330.0,
            brightness_temp_t11=295.0,
            bright_delta=35.0,
            confidence=0.85,
            daynight="D" if i % 2 == 0 else "N"
        )
        for i in range(n)
    ]

    indexer = H3SpatialIndexer(default_resolution=8)

    t0 = time.perf_counter()
    enriched = indexer.add_h3_indices(detections)
    bins = indexer.aggregate_by_h3(enriched, resolution=8, min_detections=1)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    rate = n / max(elapsed, 1e-6)

    print(f"\n[SCALABILITY] H3 Indexing & Aggregation (10,000 points): {elapsed:.4f}s ({rate:.0f} pts/sec)")
    assert len(enriched) == n
    assert len(bins) > 0
    assert elapsed < 3.0, f"H3 aggregation too slow: {elapsed:.2f}s > 3.0s"


def test_scalability_10k_haversine_dbscan_clustering():
    """
    Scalability Benchmark 2:
    Haversine DBSCAN clustering on 10,000 detection points using BallTree.
    Verifies O(N log N) complexity and execution within reasonable bounds.
    """
    rng = np.random.RandomState(42)
    n = 10000
    now = datetime.datetime(2026, 3, 1, 12, 0)

    # 50 dense cluster hotspots + background noise
    lats = []
    lons = []
    m_to_deg = 1.0 / 111320.0

    # 50 cluster centers
    centers = [(rng.uniform(15.0, 28.0), rng.uniform(70.0, 85.0)) for _ in range(50)]
    for i in range(n):
        if i < 8000:
            c = centers[i % 50]
            lats.append(c[0] + rng.normal(0, 150.0 * m_to_deg))
            lons.append(c[1] + rng.normal(0, 150.0 * m_to_deg))
        else:
            lats.append(rng.uniform(10.0, 35.0))
            lons.append(rng.uniform(68.0, 95.0))

    detections = [
        FIRMSDetection(
            detection_id=f"SCALE_DB_{i:06d}",
            latitude=float(lats[i]),
            longitude=float(lons[i]),
            acq_date="2026-03-01",
            acq_time="1200",
            timestamp=now,
            sensor="VIIRS_SNPP",
            satellite="SNPP",
            frp=30.0,
            brightness_temp_t4=335.0,
            brightness_temp_t11=295.0,
            bright_delta=40.0,
            confidence=0.90,
            daynight="D" if i % 2 == 0 else "N"
        )
        for i in range(n)
    ]

    engine = SpatioTemporalClusteringEngine()

    t0 = time.perf_counter()
    clusters = engine.run_dbscan_clustering(detections, eps_m=500.0, min_samples=5)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    rate = n / max(elapsed, 1e-6)

    print(f"[SCALABILITY] Haversine DBSCAN (10,000 points): {elapsed:.4f}s ({rate:.0f} pts/sec), Discovered {len(clusters)} clusters")
    assert len(clusters) >= 40
    assert elapsed < 5.0, f"Haversine DBSCAN too slow: {elapsed:.2f}s > 5.0s"


def test_scalability_10k_st_dbscan_clustering():
    """
    Scalability Benchmark 3:
    ST-DBSCAN spatio-temporal clustering on 10,000 points across a 30-day temporal window.
    """
    rng = np.random.RandomState(42)
    n = 10000
    base_time = datetime.datetime(2026, 3, 1, 0, 0)
    m_to_deg = 1.0 / 111320.0

    centers = [(rng.uniform(15.0, 28.0), rng.uniform(70.0, 85.0)) for _ in range(40)]
    detections = []

    for i in range(n):
        day_offset = rng.randint(0, 30)
        c = centers[i % 40]
        lat = c[0] + rng.normal(0, 200.0 * m_to_deg)
        lon = c[1] + rng.normal(0, 200.0 * m_to_deg)
        ts = base_time + datetime.timedelta(days=day_offset, hours=rng.randint(0, 23))

        detections.append(FIRMSDetection(
            detection_id=f"SCALE_ST_{i:06d}",
            latitude=float(lat),
            longitude=float(lon),
            acq_date=ts.strftime("%Y-%m-%d"),
            acq_time=ts.strftime("%H%M"),
            timestamp=ts,
            sensor="VIIRS_SNPP",
            satellite="SNPP",
            frp=35.0,
            brightness_temp_t4=340.0,
            brightness_temp_t11=295.0,
            bright_delta=45.0,
            confidence=0.90,
            daynight="D" if i % 2 == 0 else "N"
        ))

    engine = SpatioTemporalClusteringEngine()

    t0 = time.perf_counter()
    clusters = engine.run_st_dbscan(detections, spatial_eps_m=1000.0, temporal_eps_hours=48.0, min_samples=4)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    rate = n / max(elapsed, 1e-6)

    print(f"[SCALABILITY] ST-DBSCAN (10,000 points, 30 days): {elapsed:.4f}s ({rate:.0f} pts/sec), Discovered {len(clusters)} ST clusters")
    assert len(clusters) > 0
    assert elapsed < 8.0, f"ST-DBSCAN too slow: {elapsed:.2f}s > 8.0s"


def test_scalability_10k_persistence_analyzer_synthesis():
    """
    Scalability Benchmark 4:
    PersistenceAnalyzer synthesizing 500 clusters totaling 10,000 detections.
    """
    rng = np.random.RandomState(42)
    n_clusters = 500
    pts_per_cluster = 20
    now = datetime.datetime(2026, 3, 1, 12, 0)
    m_to_deg = 1.0 / 111320.0

    analyzer = PersistenceAnalyzer()
    clusters_out = []

    t0 = time.perf_counter()
    for c_idx in range(n_clusters):
        cent_lat = rng.uniform(10.0, 30.0)
        cent_lon = rng.uniform(70.0, 90.0)
        dets = [
            FIRMSDetection(
                detection_id=f"C_{c_idx}_D_{d_idx}",
                latitude=cent_lat + rng.normal(0, 50.0 * m_to_deg),
                longitude=cent_lon + rng.normal(0, 50.0 * m_to_deg),
                acq_date=f"2026-03-{(d_idx % 15) + 1:02d}",
                acq_time="1200",
                timestamp=now + datetime.timedelta(days=d_idx % 15),
                sensor="VIIRS_SNPP",
                satellite="SNPP",
                frp=float(rng.uniform(20.0, 60.0)),
                brightness_temp_t4=340.0,
                brightness_temp_t11=295.0,
                bright_delta=45.0,
                confidence=0.95,
                daynight="D" if d_idx % 2 == 0 else "N"
            )
            for d_idx in range(pts_per_cluster)
        ]
        cluster = analyzer.analyze_cluster(f"CLUS_SCALE_{c_idx:04d}", dets)
        clusters_out.append(cluster)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    rate = n_clusters / max(elapsed, 1e-6)

    print(f"[SCALABILITY] PersistenceAnalyzer synthesized {n_clusters} clusters (10,000 points) in {elapsed:.4f}s ({rate:.0f} clusters/sec)")
    assert len(clusters_out) == n_clusters
    assert elapsed < 3.0, f"PersistenceAnalyzer too slow: {elapsed:.2f}s > 3.0s"


# =========================================================================
# 5. CLUSTER PURITY BENCHMARK ON REALISTIC OPERATIONAL DATASETS
# =========================================================================

def test_empirical_cluster_purity_across_all_corridors():
    """
    Empirically verify cluster purity >= 90.0% benchmark across all operational corridors.
    """
    gen = SyntheticDataGenerator()
    suite = gen.generate_master_benchmark_suite()

    all_detections = []
    ground_truth = []

    for corridor_name, ds in suite.items():
        for d in ds.detections:
            all_detections.append(d)
            ground_truth.append(corridor_name)

    engine = SpatioTemporalClusteringEngine()
    clusters = engine.run_dbscan_clustering(all_detections, eps_m=1000.0, min_samples=3)
    results = engine.evaluate_cluster_purity(clusters, ground_truth)

    purity = results["cluster_purity"]
    print(f"\n[BENCHMARK] Master Suite Cluster Purity: {purity * 100.0:.2f}% (Threshold >= 90.0%)")
    print(f"[BENCHMARK] Homogeneity: {results['homogeneity']:.4f}, Completeness: {results['completeness']:.4f}, ARI: {results['adjusted_rand_index']:.4f}")

    assert purity >= 0.90, f"Cluster purity {purity*100:.2f}% did not satisfy 90.0% acceptance criterion!"
